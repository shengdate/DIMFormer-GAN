import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import DeformConv2d as DECNv2

#Deformable Convolution Block
class DCNv2(nn.Module):
    def __init__(self,n_chan,n_class,k = 3, s = 1, padding = None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.offset_conv = nn.Conv2d(n_chan, 2 * k * k, kernel_size=k, stride=s, padding=padding)
        self.deform_conv = DECNv2(n_chan,n_class, kernel_size=k, stride=s, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_class)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        out = self.bn(out)
        out = self.relu(out)
        return out

#Residual convolution block
class ResBlock(nn.Module):
    def __init__(self,chan_in,chan_out,stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(chan_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(chan_out, chan_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chan_out),
        )
        if chan_in != chan_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(chan_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = nn.ReLU(inplace=True)(out)
        return out

#Basic convolution layer
class ConvBNA(nn.Module):
    def __init__(self,n_chan,n_class,k,s,padding=None):
        super().__init__()
        if padding is None:
            padding = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(n_chan, n_class, kernel_size=k, stride=s, padding=padding,bias=False),
            nn.BatchNorm2d(n_class),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out = self.net(x)
        return out

#Build a multilayer perceptron
class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_layers =3 ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            a = in_dim if i == 0 else hidden_dim
            b = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(a, b))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out

#Four-stage backbone network (CNN structure)
class Simpleblock(nn.Module):
    def __init__(self, in_channels = 3, widths = (64,128,256,512) ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNA(in_channels, widths[0],3,2),
            ResBlock(widths[0], widths[0],1),
            ResBlock(widths[0], widths[0],2),
        )
        self.stage3 = nn.Sequential(
            ResBlock(widths[0], widths[1],2),
            ResBlock(widths[1], widths[1],1),
        )
        self.stage4 = nn.Sequential(
            ResBlock(widths[1], widths[2],2),
            DCNv2(widths[2], widths[2],3,1),
        )
        self.stage5 = nn.Sequential(
            ResBlock(widths[2], widths[3],2),
            DCNv2(widths[3], widths[3],3,1),
        )
    def forward(self, x):
        c2 = self.stem(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'c5': c5,
        }

#Pixel decoder
class PixelBlock(nn.Module):
    def __init__(self,in_channels = (64,128,256,512), hidden_dim=128, mask_dim=128, ):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, kernel_size=1)
            for c in in_channels
        ])
        self.output = nn.ModuleList([
            ConvBNA(hidden_dim, hidden_dim, 3,1)
            for _ in in_channels
                                     ])
        self.mask_feature = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
    def forward(self, feats):
        xs = [
            feats["c2"],
            feats["c3"],
            feats["c4"],
            feats["c5"],
        ]
        ps = [None] * len(xs)
        last_inner = None
        for i in reversed(range(len(xs))):
            lateral = self.lateral[i](xs[i])
            if last_inner is None:
                inner = lateral
            else:
                inner = lateral + F.interpolate(last_inner, size=lateral.shape[-2:], mode='nearest',)
            ps[i] = self.output[i](inner)
            last_inner = inner

        mask_feature = self.mask_feature(ps[0])
        multi_scale = [
            ps[3],
            ps[2],
            ps[1],
            ps[0],
        ]
        return mask_feature, multi_scale

#Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self,num_pos = 64, temperature = 10000,normalize = True, scale = 2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
    def forward(self, x):
        b, _, h, w = x.shape
        y_embed = torch.arange(h, device=x.device).float()
        x_embed = torch.arange(w, device=x.device).float()
        y_embed = y_embed.unsqueeze(1).repeat(1, w)
        x_embed = x_embed.unsqueeze(0).repeat(h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h - 1 + eps) * self.scale
            x_embed = x_embed / (w - 1 + eps) * self.scale
        dim_t = torch.arange( self.num_pos_feats, device=x.device, dtype=torch.float32,)
        dim_t = self.temperature ** ( 2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(( pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos(),), dim=3,).flatten(2)
        pos_y = torch.stack(( pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos(),), dim=3,).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos

#Transform decoder layer
class MaskeDecoderLayer(nn.Module):
    def __init__(self,d_model=128,nhead=8,dim_feedforward=512):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention( embed_dim=d_model, num_heads=nhead, batch_first=True,)
        self.self_attn = nn.MultiheadAttention( embed_dim=d_model, num_heads=nhead, batch_first=True,)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)
    def forward(self, tgt, memory, query_pos, pos, attn_mask = None ):
        q = tgt + query_pos
        k = memory + pos
        tgt2, _ = self.cross_attn( query=q, key=k, value=memory, attn_mask=attn_mask,)
        tgt = self.norm1(tgt + tgt2)
        q = tgt + query_pos
        k = tgt + query_pos
        tgt2, _ = self.self_attn( query=q, key=k, value=tgt,)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(self.act(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt

#mask generator
class Mask2Encoder(nn.Module):
    def __init__(self, num_queries=6, num_classes=1, hidden_dim=128, mask_dim=128, nheads=8, num_layers=8,):
        super().__init__()
        self.num_queries = num_queries
        self.nheads = nheads
        self.num_layers = num_layers
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_embed = PositionalEncoding( num_pos=hidden_dim // 2,)
        self.layers = nn.ModuleList([
            MaskeDecoderLayer( d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4,)
            for _ in range(num_layers)
        ])
        self.class_embed = nn.Linear( hidden_dim, num_classes + 1,)
        self.mask_embed = MLP( in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=mask_dim, num_layers=3,)
    def prediction_heads(self, query, mask_features):
        class_logits = self.class_embed(query)
        mask_embed = self.mask_embed(query)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embed,
            mask_features,
        )
        return class_logits, mask_logits

    def build_attention_mask(self, mask_logits, target_hw):
        mask = F.interpolate( mask_logits, size=target_hw, mode="bilinear", align_corners=False,)
        attn_mask = mask.sigmoid().flatten(2) < 0.5
        all_masked = attn_mask.all(dim=-1, keepdim=True)
        attn_mask = torch.where(all_masked, torch.zeros_like(attn_mask), attn_mask,)
        attn_mask = attn_mask.repeat_interleave(self.nheads,dim=0,)
        return attn_mask.detach()

    def forward(self, multi_scale_features, mask_features):
        b = mask_features.shape[0]
        query = self.query_feat.weight.unsqueeze(0).repeat(b, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        pred_logits, pred_masks = self.prediction_heads(query, mask_features,)
        aux_outputs = []
        num_feature_levels = len(multi_scale_features)
        for layer_idx, layer in enumerate(self.layers):
            src_2d = multi_scale_features[layer_idx % num_feature_levels]
            _, _, h, w = src_2d.shape
            memory = src_2d.flatten(2).transpose(1, 2)
            pos = self.pos_embed(src_2d)
            pos = pos.flatten(2).transpose(1, 2)
            attn_mask = self.build_attention_mask(pred_masks, target_hw=(h, w),)
            query = layer(tgt=query, memory=memory, query_pos=query_pos, pos=pos, attn_mask=attn_mask,)
            pred_logits, pred_masks = self.prediction_heads(query,mask_features,)
            if layer_idx != self.num_layers - 1:
                aux_outputs.append({
                    "pred_logits": pred_logits,
                    "pred_masks": pred_masks,
                })
        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "aux_outputs": aux_outputs,
        }

#Mask2Former-GAN generator
class Mask2FormerGANGenerator(nn.Module):
    def __init__( self, in_channels=3, num_queries=6, hidden_dim=128, mask_dim=128, num_classes=1, num_decoder_layers=8,):
        super().__init__()
        self.backbone = Simpleblock( in_channels=in_channels, widths=(64, 128, 256, 512),)
        self.pixel_decoder = PixelBlock( in_channels=(64, 128, 256, 512), hidden_dim=hidden_dim, mask_dim=mask_dim,)
        self.transformer_decoder = Mask2Encoder( num_queries=num_queries, num_classes=num_classes, hidden_dim=hidden_dim, mask_dim=mask_dim, nheads=8, num_layers=num_decoder_layers, )
        # Semantic segmentation branch
        self.semantic_head = nn.Sequential( ConvBNA(mask_dim, mask_dim, k=3, s=1), nn.Conv2d(mask_dim, 1, kernel_size=1),)
    def forward(self, x):
        image_hw = x.shape[-2:]
        feats = self.backbone(x)
        mask_features, multi_scale_features = self.pixel_decoder(feats)
        dec_out = self.transformer_decoder( multi_scale_features=multi_scale_features, mask_features=mask_features,)
        pred_masks = F.interpolate( dec_out["pred_masks"], size=image_hw, mode="bilinear", align_corners=False,)
        semantic_logits = self.semantic_head(mask_features)
        semantic_logits = F.interpolate( semantic_logits, size=image_hw, mode="bilinear", align_corners=False,)
        aux_outputs = []
        for aux in dec_out["aux_outputs"]:
            aux_masks = F.interpolate(
                aux["pred_masks"],
                size=image_hw,
                mode="bilinear",
                align_corners=False,
            )
            aux_outputs.append({
                "pred_logits": aux["pred_logits"],
                "pred_masks": aux_masks,
            })
        return {
            "pred_semantic_logits": semantic_logits,
            "pred_logits": dec_out["pred_logits"],
            "pred_masks": pred_masks,
            "aux_outputs": aux_outputs,
        }

#Basic discriminator convolution block
class DissConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2, use_norm = True):
        super().__init__()
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1 , bias = not use_norm),
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

#Conditional discriminator
class Discriminator(nn.Module):
    def __init__(self,  image_channels=3, mask_channels=2, base_channels=64):
        super().__init__()
        in_channels = image_channels + mask_channels
        self.net = nn.Sequential(
            DissConvBlock(in_channels, base_channels, stride = 2, use_norm = False),
            DissConvBlock(base_channels, base_channels * 2, stride = 2, use_norm = True),
            DissConvBlock(base_channels * 2, base_channels * 4, stride = 2, use_norm = True),
            DissConvBlock(base_channels * 4, base_channels * 8, stride = 2, use_norm = True),
            nn.utils.spectral_norm(
                nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
            )
        )
    def forward(self, image,semantic_mask, instance_mask):
        if semantic_mask.shape[-2:] != image.shape[-2:]:
            semantic_mask  = F.interpolate(semantic_mask, size=image.shape[-2:], mode="bilinear", align_corners=False,)
        if instance_mask.shape[-2:] != image.shape[-2:]:
            instance_mask = F.interpolate(instance_mask, size=image.shape[-2:], mode="nearest", )
        x = torch.cat([image, semantic_mask, instance_mask], dim=1)
        return self.net(x)


#Semantic-constrained instance module
class semationmask(nn.Module):
    def __init__(self, detach_semantic=False, eps=1e-6):
        super().__init__()
        self.detach_semantic = detach_semantic
        self.eps = eps
    def soft_union(self, masks, scores=None):
        if scores is not None:
            masks = masks * scores[:, :, None, None]
        masks = masks.clamp(self.eps, 1.0 - self.eps)
        union = 1.0 - torch.prod(1.0 - masks, dim=1, keepdim=True)
        return union.clamp(self.eps, 1.0 - self.eps)
    def forward(self, outputs):
        pred_semantic_logits = outputs["pred_semantic_logits"]
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        semantic_prob = torch.sigmoid(pred_semantic_logits)
        mask_prob = torch.sigmoid(pred_masks)
        # Class 1 = dispersive energy instance
        fg_scores = torch.softmax(pred_logits, dim=-1)[..., 1]
        if self.detach_semantic:
            sem_for_gate = semantic_prob.detach()
        else:
            sem_for_gate = semantic_prob
        # Semantically constrained instances
        constrained_masks = mask_prob * sem_for_gate
        # Union of raw instances
        raw_union = self.soft_union(masks=mask_prob,scores=fg_scores,)
        # Union of semantically constrained instances
        constrained_union = self.soft_union( masks=constrained_masks,scores=fg_scores,)
        return {
            "semantic_prob": semantic_prob,
            "fg_scores": fg_scores,
            "mask_prob": mask_prob,
            "constrained_masks": constrained_masks,
            "raw_union": raw_union,
            "constrained_union": constrained_union,
        }

# Semantic branch loss
def dice_loss_prob(prob, target, eps=1e-6):
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    loss = 1.0 - (2.0 * inter + eps) / (union + eps)
    return loss.mean()

# BCE + Dice
def semantic_branch_loss(pred_semantic_logits, sem_gt,a=0.5,b=0.5):
    loss_bce = F.binary_cross_entropy_with_logits( pred_semantic_logits,sem_gt)
    sem_prob = pred_semantic_logits.sigmoid()
    loss_dice = dice_loss_prob(sem_prob, sem_gt)
    loss_sem = a * loss_bce + b * loss_dice
    return {
        "loss_sem": loss_sem,
        "loss_sem_bce": loss_bce,
        "loss_sem_dice": loss_dice,
    }

# Mask Dice loss for matched instances
def batch_dice_cost(pred_masks, tgt_masks, eps=1e-6):
    pred = pred_masks.sigmoid().flatten(1)
    tgt = tgt_masks.flatten(1)
    numerator = 2.0 * torch.einsum("qh,nh->qn", pred, tgt)
    denominator = pred.sum(dim=1)[:, None] + tgt.sum(dim=1)[None, :]
    cost = 1.0 - (numerator + eps) / (denominator + eps)
    return cost

# Mask BCE loss for matched instances
def batch_bce_cost(pred_masks, tgt_masks):
    Q, H, W = pred_masks.shape
    N = tgt_masks.shape[0]
    pred = pred_masks.flatten(1)
    tgt = tgt_masks.flatten(1)
    pred = pred[:, None, :]
    tgt = tgt[None, :, :]
    cost = F.binary_cross_entropy_with_logits(pred.expand(Q, N, -1), tgt.expand(Q, N, -1), reduction="none").mean(dim=-1)
    return cost

#Hungarian matcher
class HungarianMatcher:
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0, match_size=(256, 256),):
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.match_size = match_size
    @torch.no_grad()
    def __call__(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        B, Q, H, W = pred_masks.shape
        indices = []
        for b in range(B):
            out_prob = pred_logits[b].softmax(dim=-1)
            out_mask = pred_masks[b]
            tgt_labels = targets[b]["labels"]
            tgt_masks = targets[b]["masks"]
            N = tgt_masks.shape[0]
            if N == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=pred_masks.device),
                    torch.empty(0, dtype=torch.long, device=pred_masks.device),
                ))
                continue
            out_mask_small = F.interpolate(out_mask[:, None], size=self.match_size, mode="bilinear", align_corners=False)[:, 0]
            tgt_mask_small = F.interpolate( tgt_masks[:, None], size=self.match_size, mode="nearest")[:, 0]
            cost_class = -out_prob[:, tgt_labels]
            cost_mask = batch_bce_cost( out_mask_small, tgt_mask_small )
            cost_dice = batch_dice_cost(out_mask_small, tgt_mask_small)
            C = (self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice)
            src_idx, tgt_idx = linear_sum_assignment(C.detach().cpu().numpy())
            src_idx = torch.as_tensor(src_idx, dtype=torch.long, device=pred_masks.device)
            tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long, device=pred_masks.device)
            indices.append((src_idx, tgt_idx))
        return indices

#Instance branch loss
def dice_loss_logits(pred_logits, targets, eps=1e-6):
    pred = pred_logits.sigmoid().flatten(1)
    tgt = targets.flatten(1)
    inter = (pred * tgt).sum(dim=1)
    union = pred.sum(dim=1) + tgt.sum(dim=1)
    loss = 1.0 - (2.0 * inter + eps) / (union + eps)
    return loss.mean()

def mask_bce_loss(pred_logits, targets):
    return F.binary_cross_entropy_with_logits( pred_logits, targets, reduction="mean" )

class InstanceCriterion(nn.Module):
    def __init__( self, matcher, num_classes=1, eos_coef=0.1,lambda_mask=5.0,lambda_dice=5.0,aux_weight=0.5,):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.lambda_mask = lambda_mask
        self.lambda_dice = lambda_dice
        self.aux_weight = aux_weight

        empty_weight = torch.ones(2)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)
    def loss_single_layer(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        B, Q, H, W = pred_masks.shape
        indices = self.matcher(outputs, targets)

        target_classes = torch.zeros( (B, Q), dtype=torch.long, device=pred_logits.device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
        loss_cls = F.cross_entropy( pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        src_masks = []
        tgt_masks = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_masks.append(pred_masks[b, src_idx])
                tgt_masks.append(targets[b]["masks"][tgt_idx])
        if len(src_masks) == 0:
            loss_mask = pred_masks.sum() * 0.0
            loss_dice = pred_masks.sum() * 0.0
        else:
            src_masks = torch.cat(src_masks, dim=0)
            tgt_masks = torch.cat(tgt_masks, dim=0)
            loss_mask = mask_bce_loss(src_masks, tgt_masks)
            loss_dice = dice_loss_logits(src_masks, tgt_masks)
        loss_ins = loss_cls + self.lambda_mask * loss_mask + self.lambda_dice * loss_dice
        return {
            "loss_ins": loss_ins,
            "loss_cls": loss_cls,
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }
    def forward(self, outputs, targets):
        main_losses = self.loss_single_layer(outputs, targets)
        pred_logits = outputs["pred_logits"]
        total_aux = pred_logits.sum() * 0.0
        aux_loss_dict = {}
        aux_outputs = outputs.get("aux_outputs", [])
        num_aux = len(aux_outputs)
        if num_aux > 0:
            aux_weights = torch.arange(1,num_aux + 1,device=pred_logits.device,dtype=pred_logits.dtype,)
            aux_weights = (aux_weights / aux_weights.sum())
            for i, (aux, weight) in enumerate(zip(aux_outputs, aux_weights)):
                aux_losses = self.loss_single_layer(aux, targets)
                weighted_aux_loss = (weight * aux_losses["loss_ins"])
                total_aux = (total_aux + weighted_aux_loss)
                aux_loss_dict[f"loss_aux_{i}"] = aux_losses["loss_ins"]
                aux_loss_dict[f"loss_aux_weighted_{i}"] = weighted_aux_loss
        main_losses["loss_aux"] = (self.aux_weight * total_aux)
        main_losses["loss_ins_total"] = (main_losses["loss_ins"] + main_losses["loss_aux"])
        main_losses.update(aux_loss_dict)
        return main_losses

# Semantic-instance consistency loss
def semantic_instance_consistency_loss( constrained_union, raw_union, semantic_prob, sem_gt,):
    loss_constrained_to_gt = dice_loss_prob(constrained_union,sem_gt)
    loss_raw_to_sem = dice_loss_prob(raw_union,semantic_prob.detach())
    loss_cons = loss_constrained_to_gt + loss_raw_to_sem
    return {
        "loss_cons": loss_cons,
        "loss_cons_gt": loss_constrained_to_gt,
        "loss_cons_sem": loss_raw_to_sem,
    }


# Discriminator loss
def discriminator_hinge_loss( D, image, fake_semantic, fake_instance_union, real_semantic, real_instance_union,):

    pred_real = D( image=image, semantic_mask=real_semantic, instance_mask=real_instance_union,)

    pred_fake = D( image=image, semantic_mask=fake_semantic.detach(), instance_mask=fake_instance_union.detach(),)
    loss_real = F.relu(1.0 - pred_real).mean()
    loss_fake = F.relu(1.0 + pred_fake).mean()
    loss_D = loss_real + loss_fake
    return {
        "loss_D": loss_D,
        "loss_D_real": loss_real,
        "loss_D_fake": loss_fake,
        "D_real_score": pred_real.mean().detach(),
        "D_fake_score": pred_fake.mean().detach(),
    }

def generator_adversarial_loss(D, image, fake_semantic, fake_instance_union,):
    pred_fake = D( image=image, semantic_mask=fake_semantic, instance_mask=fake_instance_union,)
    loss_adv_G = -pred_fake.mean()
    return {
        "loss_adv_G": loss_adv_G,
        "G_fake_score": pred_fake.mean().detach(),
    }


# Total generator loss
class GeneratorTotalLoss(nn.Module):
    def __init__(self, instance_criterion, lambda_sem=1.0, lambda_ins=1.0, lambda_cons=0.5, lambda_adv=0.01,):
        super().__init__()
        self.instance_criterion = instance_criterion
        self.lambda_sem = lambda_sem
        self.lambda_ins = lambda_ins
        self.lambda_cons = lambda_cons
        self.lambda_adv = lambda_adv
        self.semantic_constraint_for_G = semationmask( detach_semantic=False)
    def forward(self, outputs, targets, sem_gt, image, D=None, use_adv=True):
        # 1. Semantic loss
        sem_losses = semantic_branch_loss(outputs["pred_semantic_logits"],sem_gt )
        # 2. Instance branch loss, including Hungarian matching
        ins_losses = self.instance_criterion( outputs, targets)
        # 3. Semantically constrained instance results
        constraint_out = self.semantic_constraint_for_G(outputs)
        semantic_prob = constraint_out["semantic_prob"]
        raw_union = constraint_out["raw_union"]
        constrained_union = constraint_out["constrained_union"]
        # 4. Semantic-instance consistency loss
        cons_losses = semantic_instance_consistency_loss( constrained_union=constrained_union, raw_union=raw_union, semantic_prob=semantic_prob, sem_gt=sem_gt )
        # 5. Generator adversarial loss
        if use_adv and D is not None:
            adv_losses = generator_adversarial_loss( D=D, image=image, fake_semantic=semantic_prob, fake_instance_union=constrained_union, )
            loss_adv = adv_losses["loss_adv_G"]
        else:
            loss_adv = torch.tensor(0.0,device=image.device)
            adv_losses = {"loss_adv_G": loss_adv, "G_fake_score": torch.tensor( 0.0, device=image.device),}

        # 6. Total generator loss
        loss_G = (
            self.lambda_sem * sem_losses["loss_sem"]
            + self.lambda_ins * ins_losses["loss_ins_total"]
            + self.lambda_cons * cons_losses["loss_cons"]
            + self.lambda_adv * loss_adv
        )

        loss_dict = {
            "loss_G": loss_G,

            "loss_sem": sem_losses["loss_sem"],
            "loss_sem_bce": sem_losses["loss_sem_bce"],
            "loss_sem_dice": sem_losses["loss_sem_dice"],

            "loss_ins_total": ins_losses["loss_ins_total"],
            "loss_cls": ins_losses["loss_cls"],
            "loss_mask": ins_losses["loss_mask"],
            "loss_dice": ins_losses["loss_dice"],
            "loss_aux": ins_losses["loss_aux"],

            "loss_cons": cons_losses["loss_cons"],
            "loss_cons_gt": cons_losses["loss_cons_gt"],
            "loss_cons_sem": cons_losses["loss_cons_sem"],

            "loss_adv_G": loss_adv,

            "constrained_union": constrained_union.detach(),
            "raw_union": raw_union.detach(),
            "semantic_prob": semantic_prob.detach(),
        }
        loss_dict.update(adv_losses)
        return loss_dict

#Build training target from instance mask
def build_mask(instance_mask):
    if not torch.is_tensor(instance_mask):
       instance_mask = torch.as_tensor(instance_mask)
    instance_mask = instance_mask.long()
    sem_mask = (instance_mask > 0).float().unsqueeze(0)
    instance_ids = torch.unique(instance_mask)
    instance_ids = instance_ids[instance_ids > 0]
    masks  = []
    labels = []
    for ins_id in instance_ids:
        mask = (instance_mask == ins_id).float()
        masks.append(mask)
        labels.append(1)
    if len(masks) == 0:
        h, w = instance_mask.shape
        masks = torch.zeros((0, h, w), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)
    else:
        masks = torch.stack(masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
    target = {
        "masks": masks,
        "labels": labels,
        "sem_mask": sem_mask,
    }
    return target
