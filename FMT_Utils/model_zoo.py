
from FMT_Utils.FMTNoConvolution_1_1 import reject_retired_fmt, assert_no_fmt_convolution
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pnn.models.point_nn import EncNPNew
from FMT_Utils.DCT_FMT_encoder import DCT_FMT


def calculate_model_parm_size(model: nn.Module):
    """
    Summarize parameter/buffer counts and memory footprint (MB) of a PyTorch model.
    Returns a dict: param_count, trainable_param_count, buffer_count,
                    param_size_mb, buffer_size_mb, total_size_mb.
    """
    with torch.no_grad():
        param_count = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffer_count = sum(b.numel() for b in model.buffers())

        param_bytes = sum(p.numel() * (p.element_size() if p.is_floating_point() or p.dtype is not None else 4) for p in model.parameters())
        buffer_bytes = sum(b.numel() * (b.element_size() if b.is_floating_point() or b.dtype is not None else 4) for b in model.buffers())
        total_bytes = param_bytes + buffer_bytes

        mb = 1024.0 * 1024.0
        return {
            "param_count": int(param_count),
            "trainable_param_count": int(trainable_param_count),
            "buffer_count": int(buffer_count),
            "param_size_mb": float(param_bytes / mb),
            "buffer_size_mb": float(buffer_bytes / mb),
            "total_size_mb": float(total_bytes / mb),
        }

class Decoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.ln1=nn.Linear(in_dim, 256)
        self.bn1=nn.BatchNorm1d(256)
        self.ln2=nn.Linear(256, 256)
        self.bn2=nn.BatchNorm1d(256)
        self.ln3=nn.Linear(256, 64)
        self.bn3=nn.BatchNorm1d(64)
        self.ln4=nn.Linear(64, 1)

    def forward(self, x):
        x = x.contiguous()
        x = self.ln1(x.contiguous())
        x = self.bn1(x)
        x = nn.GELU()(x)


        res=x
        x = self.ln2(x)
        x = self.bn2(x)
        x = nn.GELU()(x)
        x = x + res


        x = self.ln3(x)
        x = self.bn3(x)
        x = nn.GELU()(x)

        x = self.ln4(x)
        x = nn.Sigmoid()(x)
        return x.squeeze(-1)
    
class PointWiseFMT_Regressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.encoder = EncNPNew(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)
        self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
        self.decoder = Decoder(in_dim=self.decoderInputDim)
        assert_no_fmt_convolution(self)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        PointsRaw=pts.reshape(B,CrossSize*LstepsPerline,Dim)
        points_3N=PointsRaw.permute(0, 2, 1)
        points_N3=PointsRaw
        FMT_feat = self.encoder(points_N3, points_3N)
        #feature dimension is in_dim*K(steps per line)
        pred = self.decoder(FMT_feat)
        return pred

class PointWiseMLP_Regressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.decoderInputDim=3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        # xyz = pts.permute(0, 2, 1)#B,N,3
        #feat: (B, embed_dim)
        start_pos=pts[:,:,0,:].reshape(B, -1)
        end_pos=pts[:,:,-1,:].reshape(B, -1)
        every_cross_feature=torch.cat([start_pos,end_pos],dim=1)
        pred = self.decoder(every_cross_feature)
        return pred
 



class FTLEUpsamplingFMT_Unet(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("FTLEUpsamplingFMT_Unet")

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed to handle odd sizes
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        h = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = h + attn_out

        h2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h2 + x
        return x



class UpsamplingUnetModel(nn.Module):
    """
    UNet for upsampling FTLE from lowRes to highRes
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale:float,base_ch: int = 24):
        super().__init__()
        self.upscale = int(upscale)
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)

        # Low-resolution feature output (same spatial size as input)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        # Progressive upsampling head (constructed by powers of 2)
        import math
        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        x = lowResFTLE.unsqueeze(1).contiguous()  # [B,1,X,Y]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)  # [B, base_ch, X, Y]
        x = self.out_low(x)   # [B, base_ch, X, Y]

        for blk in self.up_blocks:
            x = blk(x)

        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # Align to expected size (avoid 1-pixel misalignment due to odd/even sizes)
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred


class UpsamplingUnetModelV2(nn.Module):
    """
    Super-resolution-oriented UNet variant of UpsamplingUnetModel:
      - higher channel width (base_ch=64 vs 24)
      - ONE FEWER downsampling stage (a single 2x down instead of two)

    Rationale: pixel super-resolution needs high-frequency detail, which a deep
    encoder-decoder discards when it downsamples 32->16->8. Keeping a shallower
    (32->16) bottleneck with more channels preserves that detail. Tests whether a
    properly-sized UNet can beat the flat ESPCN CNN on this task.

    Inputs:  lowResFTLE [B, X, Y]   (pathlines ignored)
    Output:  pred       [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float, base_ch: int = 64):
        super().__init__()
        self.upscale = int(upscale)
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)        # single downsample: X -> X/2
        self.up1 = Up(base_ch * 2, base_ch)            # single upsample back to X
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        x = lowResFTLE.unsqueeze(1).contiguous()  # [B,1,X,Y]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)   # [B, base_ch, X, Y]
        x = self.out_low(x)

        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred


class AttentionFusion(nn.Module):
    """Removed with the convolutional FMT U-Net fusion models."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt('AttentionFusion')


class FTLEupsamplingFMT_UnetV3(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("FTLEupsamplingFMT_UnetV3")



class FTLEupsamplingFMT_UnetV2(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("FTLEupsamplingFMT_UnetV2")


class FTLEupsamplingDCT_FMT_UnetV2(nn.Module):
    """Removed: convolutional FMT is prohibited by project policy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        reject_retired_fmt("FTLEupsamplingDCT_FMT_UnetV2")



      
# ============================================================================
# Flow-map upsampling models
# ----------------------------------------------------------------------------
# Input/label are the cross-primitive flow map [B, P, 5, 2, 3] = (P grid cells)
# x (5 cross lines: center,x+,x-,y+,y-) x (2 endpoints: head,tail) x (3: x,y,t).
# FTLE is computed downstream from this via computeFTLEFromPathlineCrossPrimitive.
#
# Channel handling: the time channel (t) of every endpoint is CONSTANT across the
# grid (all seeds share the slice time and integration window), so BatchNorm would
# destroy it. We therefore route only the (x,y) channels (5*2*2=20) through the CNN
# and pass the t channels (5*2=10) through by nearest-neighbour resize (exact for a
# constant field). This keeps FTLE's physical time span ΔT correct.
# ============================================================================
class _FlowMapUpsamplerBase(nn.Module):
    """Shared field<->image plumbing for flow-map upsamplers. Subclasses implement
    `_cnn(xy_img)` mapping [B,20,H,W] -> [B,20,H*UP,W*UP]."""
    NERB = 5
    ENDPTS = 2
    FeactureChannel_XY = 5 * 2 * 2   # 20
    FeactureChannel_T = 5 * 2        # 10
    C_XY = FeactureChannel_XY        # alias used by subclasses (e.g. UNet_FlowMap)

    def __init__(self, upscale: int):
        super().__init__()
        self.upscale = int(upscale)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, lowResFlowMap: torch.Tensor, lowResPathlines: torch.Tensor | None = None,
                hw: tuple[int, int] | None = None) -> torch.Tensor:
        # lowResFlowMap: [B, P, 5, 2, 3]; P=H*W (square patch in training, or full grid in eval via hw)
        B, P = lowResFlowMap.shape[0], lowResFlowMap.shape[1]
        if hw is None:
            side = int(round(P ** 0.5))
            assert side * side == P, f"non-square P={P}; pass hw=(H,W) explicitly"
            H, W = side, side
        else:
            H, W = int(hw[0]), int(hw[1])
        UP = max(1, self.upscale)
        Hh, Wh = H * UP, W * UP

        fm = lowResFlowMap.reshape(B, H, W, self.NERB, self.ENDPTS, 3)
        xy = fm[..., :2]   # [B,H,W,5,2,2]
        t  = fm[..., 2:]   # [B,H,W,5,2,1]
        xy_img = xy.reshape(B, H, W, self.FeactureChannel_XY).permute(0, 3, 1, 2).contiguous()  # [B,20,H,W]
        t_img  = t.reshape(B, H, W, self.FeactureChannel_T).permute(0, 3, 1, 2).contiguous()    # [B,10,H,W]
        xy_hi = self._cnn(xy_img)  # [B,20,Hh,Wh]
        if xy_hi.shape[-2] != Hh or xy_hi.shape[-1] != Wh:
            xy_hi = F.interpolate(xy_hi, size=(Hh, Wh), mode='bilinear', align_corners=False)
        # t is (piecewise) constant -> nearest resize reproduces it exactly
        t_hi = F.interpolate(t_img, size=(Hh, Wh), mode='nearest')  # [B,10,Hh,Wh]

        xy_hi_s = xy_hi.permute(0, 2, 3, 1).reshape(B, Hh, Wh, self.NERB, self.ENDPTS, 2)
        t_hi_s  = t_hi.permute(0, 2, 3, 1).reshape(B, Hh, Wh, self.NERB, self.ENDPTS, 1)
        out = torch.cat([xy_hi_s, t_hi_s], dim=-1)  # [B,Hh,Wh,5,2,3]
        return out.reshape(B, Hh * Wh, self.NERB, self.ENDPTS, 3)


class ESPCN_FlowMap(_FlowMapUpsamplerBase):
    """ESPCN-style flat CNN for flow-map upsampling (20-channel in/out)."""
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 2):
        super().__init__(upscale)
        f1 = int(getattr(cfg.model, 'f1', 5))
        f2 = int(getattr(cfg.model, 'f2', 3))
        n1 = int(getattr(cfg.model, 'n1', 64))
        n2 = int(getattr(cfg.model, 'n2', 32))
        p1, p2 = f1 // 2, f2 // 2
        self.conv1 = nn.Conv2d(self.FeactureChannel_XY, n1, kernel_size=f1, padding=p1, bias=True)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=p2, bias=True)
        k_up = max(2, 2 * self.upscale)
        s_up = max(1, self.upscale)
        p_up = self.upscale // 2
        self.deconv = nn.ConvTranspose2d(n2, self.FeactureChannel_XY, kernel_size=k_up, stride=s_up, padding=p_up, bias=True)
        self.act = nn.ReLU(inplace=True)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(xy_img))
        x = self.act(self.conv2(x))
        return self.deconv(x)


class UNet_FlowMap(_FlowMapUpsamplerBase):
    """SR-oriented shallow UNet (one downsample) for flow-map upsampling (20-ch in/out).
    Mirrors UpsamplingUnetModelV2 but multi-channel and with no t through BatchNorm."""
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 2, base_ch: int = 64):
        super().__init__(upscale)
        self.inc = DoubleConv(self.C_XY, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, self.C_XY, kernel_size=1)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(xy_img)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        return self.out_high(x)


class ESPCN_SR(nn.Module):
    """Canonical efficient sub-pixel CNN (Shi et al. 2016) for 2-channel flow-map SR,
    matching Jakob et al. 2020: two conv layers (f1,f2; n1,n2 features, ReLU) followed by
    a sub-pixel convolution that upsamples by `upscale`. Input/output are 2-channel images
    (the particle end-position flow map). Fully convolutional -> any H x W at inference."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2):
        super().__init__()
        f1 = int(getattr(cfg.model, 'f1', 3))
        f2 = int(getattr(cfg.model, 'f2', 3))
        n1 = int(getattr(cfg.model, 'n1', 128))
        n2 = int(getattr(cfg.model, 'n2', 128))
        self.upscale = int(upscale)
        self.in_ch = int(in_ch)
        self.conv1 = nn.Conv2d(in_ch, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(self.upscale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W] -> [B, 2, H*UP, W*UP]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return self.shuffle(x)


class UNet_SR(nn.Module):
    """UNet body (2 downsamples) + sub-pixel upsampling head for 2-channel flow-map SR.
    Same I/O contract as ESPCN_SR (2-ch image -> 2-ch image at k x). Fully convolutional."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2, base: int = 64):
        super().__init__()
        base = int(getattr(cfg.model, 'base', base))
        self.upscale = int(upscale)
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.up1 = Up(base * 4, base * 2)
        self.up2 = Up(base * 2, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1),
            nn.PixelShuffle(self.upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.head(x)


def build_model(config, device):
    nerb = int(config.pcds.num_cross_points_per_seeding)
    L = int(config.pcds.sampled_points_per_line)
    if config.model.NAME == 'FMT_Regressor':
        model = PointWiseFMT_Regressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
    elif config.model.NAME == 'MLP_Regressor':
        model = PointWiseMLP_Regressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
        return model
    elif config.model.NAME == 'UpsamplingUnetModel':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UpsamplingUnetModel(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'UpsamplingUnetModelV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UpsamplingUnetModelV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_Unet':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingFMT_Unet(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_UnetV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingFMT_UnetV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingDCT_FMT_UnetV2' or config.model.NAME == 'DCT_FMT_UnetV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingDCT_FMT_UnetV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_UnetV3' or config.model.NAME == 'FTLEUpsamplingFMT_Unet_V3':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingFMT_UnetV3(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ESPCN_FlowMap':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = ESPCN_FlowMap(config, lowResX, lowResY, upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'UNet_FlowMap':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UNet_FlowMap(config, lowResX, lowResY, upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ESPCN_SR':
        model = ESPCN_SR(config, upscale=int(config.dataset.UPsampling), in_ch=2).to(device)
        return model
    elif config.model.NAME == 'UNet_SR':
        model = UNet_SR(config, upscale=int(config.dataset.UPsampling), in_ch=2).to(device)
        return model
    else:
        raise ValueError(f"Unknown model: {config.model.NAME}")
