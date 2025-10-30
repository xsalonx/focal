import torch
from torch import nn
from nflows.transforms.base import Transform

class HouseholderMix(Transform):
    """
    K reflektorów Householdera H_i = I - 2 u_i u_i^T, gdzie u_i = v_i / ||v_i||.
    Działa na wejściu [B, D]. Jest ortogonalny (|det|=1) => logabsdet = 0.
    """
    def __init__(self, features: int, num_reflections: int = 4, eps: float = 1e-8):
        super().__init__()
        self.features = int(features)
        self.num_reflections = int(num_reflections)
        self.eps = float(eps)
        # Parametry: v_i w R^D
        self.v = nn.Parameter(torch.randn(num_reflections, features) * 0.02)

    def _apply_one(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x: [B, D], v: [D]
        # u = v / ||v||, y = x - 2 * (x·u) u
        u = v / (v.norm(p=2) + self.eps)
        proj = (x @ u)                     # [B]
        y = x - 2.0 * proj.unsqueeze(1) * u.unsqueeze(0)
        return y

    def forward(self, inputs: torch.Tensor, context=None):
        y = inputs
        # H_k ... H_1 (dowolna kolejność jest ok, ale trzymajmy stałą)
        for i in range(self.num_reflections):
            y = self._apply_one(y, self.v[i])
        # log|det| = 0 dla ortogonalnych transformacji (stałe ±1)
        logabsdet = inputs.new_zeros(inputs.size(0))
        return y, logabsdet

    def inverse(self, inputs: torch.Tensor, context=None):
        y = inputs
        # (H_k ... H_1)^{-1} = H_1 ... H_k  (odwrotna kolejność!)
        for i in reversed(range(self.num_reflections)):
            y = self._apply_one(y, self.v[i])
        logabsdet = inputs.new_zeros(inputs.size(0))
        return y, logabsdet



class Squeeze2x2Vec(Transform):
    """
    Squeeze 2x2 działający na wektorach [B, D] przez reshape do [B,C,H,W].
    Zakłada parzyste H,W. D po operacji pozostaje takie samo.
    """
    def __init__(self, C: int, H: int, W: int):
        super().__init__()
        assert H % 2 == 0 and W % 2 == 0, "Squeeze2x2 wymaga parzystych H i W."
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.C2, self.H2, self.W2 = C * 4, H // 2, W // 2
        self.D = C * H * W

    def _to_image(self, x):
        B = x.size(0)
        return x.view(B, self.C, self.H, self.W)

    def _to_vec(self, x):
        B = x.size(0)
        return x.view(B, -1)

    def forward(self, inputs: torch.Tensor, context=None):
        x = self._to_image(inputs)
        # pixel-unshuffle 2x2: [B,C,H,W] -> [B,4C,H/2,W/2]
        B, C, H, W = x.shape
        y = x.view(B, C, self.H2, 2, self.W2, 2) \
             .permute(0, 1, 3, 5, 2, 4) \
             .reshape(B, self.C2, self.H2, self.W2)
        y = self._to_vec(y)
        logabsdet = inputs.new_zeros(inputs.size(0))
        return y, logabsdet

    def inverse(self, inputs: torch.Tensor, context=None):
        y = inputs.view(inputs.size(0), self.C2, self.H2, self.W2)
        B = y.size(0)
        # pixel-shuffle 2x2: [B,4C,H/2,W/2] -> [B,C,H,W]
        y = y.view(B, self.C, 2, 2, self.H2, self.W2) \
             .permute(0, 1, 4, 2, 5, 3) \
             .reshape(B, self.C, self.H, self.W)
        x = y.view(B, -1)
        logabsdet = inputs.new_zeros(inputs.size(0))
        return x, logabsdet



import torch.nn.functional as F

class InvertibleConv1x1LU(Transform):
    """
    Glow-style invertible 1x1 conv z parametryzacją LU.
    Działa na wektorach [B, D] przez reshape do [B,C,H,W].
    log|det| = (H*W) * sum(s). Brak .item() → zachowany gradient!
    """
    def __init__(self, C: int, H: int, W: int, use_permutation: bool = True, eps: float = 1e-8):
        super().__init__()
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.eps = float(eps)

        # Stała permutacja P (na bufferach → przeniesie się z .to(device))
        if use_permutation:
            perm = torch.randperm(self.C).tolist()
        else:
            perm = list(range(self.C))
        inv_perm = [0] * self.C
        for i, p in enumerate(perm): inv_perm[p] = i
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long))
        self.register_buffer("inv_perm", torch.tensor(inv_perm, dtype=torch.long))

        # Parametry L (strictly lower), U (strictly upper) i log-diagonal s
        self.L = nn.Parameter(torch.zeros(self.C, self.C))
        self.U = nn.Parameter(torch.zeros(self.C, self.C))
        self.s = nn.Parameter(torch.zeros(self.C))  # diag = exp(s) > 0

        # Maski trójkątne (nietrenowalne)
        self.register_buffer("lower_mask", torch.tril(torch.ones(self.C, self.C), -1))
        self.register_buffer("upper_mask", torch.triu(torch.ones(self.C, self.C), +1))

    def _mat_parts(self):
        # Składamy W = P @ (I + strictly_lower(L)) @ (diag(exp(s)) + strictly_upper(U))
        I = torch.eye(self.C, device=self.s.device, dtype=self.s.dtype)
        L = self.L * self.lower_mask           # strictly lower
        U = self.U * self.upper_mask           # strictly upper
        D = torch.diag(torch.exp(self.s))      # dodatnia diagonala
        A = I + L                               # lower, diag==1
        B = D + U                               # upper, diag>0
        return A, B

    def _weight(self):
        A, B = self._mat_parts()
        W = A @ B                               # lower @ upper
        W = W[self.perm, :]                     # P @ W
        return W

    def forward(self, inputs: torch.Tensor, context=None):
        B = inputs.size(0)
        x = inputs.view(B, self.C, self.H, self.W)   # [B,C,H,W]

        # y = (P @ A @ B) x   → zastosujemy jako conv2d 1x1
        W = self._weight()                           # [C,C]
        weight = W.view(self.C, self.C, 1, 1)
        y = F.conv2d(x, weight)                      # [B,C,H,W]

        out = y.reshape(B, -1)

        # log|det| = (H*W)*sum(s)  — bez .item(), z gradientem
        ld = self.s.sum() * float(self.H * self.W)   # skalar z grad
        logabsdet = ld.expand(B).to(inputs.dtype)    # [B]
        return out, logabsdet

    def inverse(self, inputs: torch.Tensor, context=None):
        B = inputs.size(0)
        y = inputs.view(B, self.C, self.H, self.W)   # [B,C,H,W]

        # Odwrót: x = B^{-1} A^{-1} P^T y
        # 1) P^T
        z = y[:, self.inv_perm, :, :]                # [B,C,H,W]

        # 2) Rozwiąż A x1 = z  (A = I+L, lower unitriangular)
        A, Bmat = self._mat_parts()                  # A: lower(1), Bmat: upper(diag>0)
        # Przepisz do macierzy [B*H*W, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.C)   # [Npix, C]
        # solve A x1 = z_flat^T
        x1 = torch.linalg.solve_triangular(A, z_flat.T, upper=False, unitriangular=True).T  # [Npix, C]

        # 3) Rozwiąż Bmat x = x1  (upper-triangular)
        x2 = torch.linalg.solve_triangular(Bmat, x1.T, upper=True).T    # [Npix, C]

        x = x2.view(B, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
        out = x.view(B, -1)

        # log|det^{-1}| = -(H*W)*sum(s)
        ld = -self.s.sum() * float(self.H * self.W)
        logabsdet = ld.expand(B).to(inputs.dtype)
        return out, logabsdet
