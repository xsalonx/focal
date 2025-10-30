import os

import torch

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from focal.utils.torch import get_data_loader, torch_tensor_to_numpy_image



import numpy as np
from typing import Optional, Tuple, Union

ArrayLike = Union[np.ndarray, "numpy.typing.NDArray[np.float32]"]

class CaloINNPreprocessorNP:
    """
    Preprocessing (NumPy):
      x  -> (opcjonalnie: x + U(0,b) w trybie train) -> log(x + α)

    Parametry:
      b_noise    : amplituda szumu U(0,b) (np. 5e-6)
      alpha_log  : dodatek pod log (np. 1e-8)
      flatten    : jeśli True -> transform zwraca [N, D]; inverse odtwarza [N,1,H,W]
      clamp_min  : minimalna wartość przed logiem (np. 0.0)
      dtype      : docelowy typ (np. np.float32)
      random_state : int lub Generator; kontrola losowości szumu

    API:
      fit(X)                -> zapamiętuje kształt (C,H,W) dla inverse
      transform(X)          -> przekształcenie; w train() dodaje szum, w eval() nie
      inverse_transform(Y)  -> exp(Y) - α (clamp do 0), ewentualny reshape do [N,1,H,W]
      train() / eval()      -> przełączanie trybu (czy dodawać szum)
    """

    def __init__(self,
                 b_noise: float = 5e-6,
                 alpha_log: float = 1e-8,
                 flatten: bool = False,
                 clamp_min: float = 0.0,
                 dtype=np.float32,
                 random_state: Optional[Union[int, np.random.Generator]] = None,
                 zscore=False,
        ):
        self.b_noise = float(b_noise)
        self.alpha_log = float(alpha_log)
        self.flatten = bool(flatten)
        self.clamp_min = float(clamp_min)
        self.dtype = dtype

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        elif isinstance(random_state, int):
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

        self.zscore = zscore
        self.mu = None
        self.std = None

        self.fitted_ = False
        self.input_shape_: Optional[Tuple[int, int, int]] = None  # (C,H,W)
        self.mode_ = "train"  # "train" (dodaje szum) / "eval" (bez szumu)

    # ---- tryb pracy ----
    def train(self):
        self.mode_ = "train"
        return self

    def eval(self):
        self.mode_ = "eval"
        return self

    # ---- core API ----
    def fit(self, X: ArrayLike):
        X = self._as_array(X)
        if X.ndim == 4:
            _, C, H, W = X.shape
            self.input_shape_ = (C, H, W)
        if X.ndim == 3:
            _, H, W = X.shape
            self.input_shape_ = (1, H, W)
        else:
            raise ValueError(f"Expected [N,1,H,W] or [N,D], got {X.shape}")

        if self.zscore:
            xlog = np.log(X + self.alpha_log)
            self.mu = xlog.mean()
            self.std = xlog.std()

        if (X < -1e-12).any():
            raise ValueError("Preprocessing zakłada wartości >= 0.")
        self.fitted_ = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted()
        x = self._as_array(X)

        # opcjonalne spłaszczenie
        if self.flatten:
            N = x.shape[0]
            x = x.reshape(N, -1)

        # clamp i (ew.) szum w trybie train
        if self.clamp_min is not None:
            x = np.maximum(x, self.clamp_min, dtype=self.dtype)

        if self.mode_ == "train" and self.b_noise > 0.0:
            x = x + self.rng.random(x.shape, dtype=self.dtype) * self.b_noise

        # log(x + α)
        if self.alpha_log > 0.0:
            x = np.log(x + self.alpha_log, dtype=self.dtype)
        else:
            x = np.log(x, dtype=self.dtype)

        if self.zscore:
            x -= self.mu
            x /= self.std

        return x.astype(self.dtype, copy=False)

    def inverse_transform(self, Y: ArrayLike, output_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        self._check_fitted()
        y = self._as_array(Y)

        if self.zscore:
            y *= self.std
            y += self.mu

        x = np.exp(y, dtype=self.dtype) - self.alpha_log
        x = np.maximum(x, 0.0, dtype=self.dtype)

        if self.flatten:
            if output_shape is not None:
                x = x.reshape((x.shape[0],) + tuple(output_shape))
            elif self.input_shape_ is not None:
                C, H, W = self.input_shape_
                x = x.reshape((x.shape[0], C, H, W))
            # jeśli brak output_shape i input_shape_, zostawiamy [N,D]
        return x.astype(self.dtype, copy=False)
    
    def deflatten(self, Y):
        return Y.reshape((Y.shape[0],) + self.input_shape_)

    # ---- utils ----
    def _as_array(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, np.ndarray):
            if X.dtype != self.dtype:
                return X.astype(self.dtype, copy=False)
            return X
        # w razie list itp.
        return np.asarray(X, dtype=self.dtype)

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Preprocessor is not fitted yet. Call fit(X) first.")


class CaloINNPreprocessorNPSimple:
    def __init__(self,
                 flatten: bool = False,
                 dtype=np.float32,
):
        self.flatten = bool(flatten)
        self.dtype = dtype
        self.alpha_log = 1e-8

        self.fitted_ = False
        self.input_shape_: Optional[Tuple[int, int, int]] = None  # (C,H,W)

    # ---- core API ----
    def fit(self, X: ArrayLike):
        X = self._as_array(X)
        if X.ndim == 4:
            _, C, H, W = X.shape
            self.input_shape_ = (C, H, W)
        if X.ndim == 3:
            _, H, W = X.shape
            self.input_shape_ = (1, H, W)
        else:
            raise ValueError(f"Expected [N,1,H,W] or [N,D], got {X.shape}")
        
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        x = self._as_array(X)

        # opcjonalne spłaszczenie
        if self.flatten:
            N = x.shape[0]
            x = x.reshape(N, -1)

        x = np.log(x + self.alpha_log, dtype=self.dtype)

        return x.astype(self.dtype, copy=False)

    def inverse_transform(self, Y: ArrayLike, output_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        y = self._as_array(Y)

        x = np.exp(y, dtype=self.dtype) - self.alpha_log

        if self.flatten:
            if output_shape is not None:
                x = x.reshape((x.shape[0],) + tuple(output_shape))
            elif self.input_shape_ is not None:
                C, H, W = self.input_shape_
                x = x.reshape((x.shape[0], C, H, W))
        return x.astype(self.dtype, copy=False)
    
    def deflatten(self, Y):
        return Y.reshape((Y.shape[0],) + self.input_shape_)

    # ---- utils ----
    def _as_array(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, np.ndarray):
            if X.dtype != self.dtype:
                return X.astype(self.dtype, copy=False)
            return X
        # w razie list itp.
        return np.asarray(X, dtype=self.dtype)



class LogPlusEpsScaler:
    def __init__(self, degree=1, eps=1):
        self.degree = degree
        self.eps = eps

    def fit(self, _x):
        pass

    def transform(self, v):
        for _ in range(self.degree):
            v = np.log(v + self.eps)
        return v
    def inverse_transform(self, v):
        for _ in range(self.degree):
            v = np.exp(v) - self.eps
        return v

class CaloINNParticlesScaler:
    def __init__(self):
        self.energy_scaler = LogPlusEpsScaler(eps=1e-12)
        self.other_scaler = StandardScaler()

    def fit(self, x):
        self.energy_scaler.fit(x[:, :1])
        self.other_scaler.fit(x[:, 1:])
        return self

    def transform(self, v):
        return np.concatenate([
            self.energy_scaler.transform(v[:, :1]),
            self.other_scaler.transform(v[:, 1:]),
        ], axis=1)

    def inverse_transform(self, v):
        return np.concatenate([
            self.energy_scaler.inverse_transform(v[:, :1]),
            self.other_scaler.inverse_transform(v[:, 1:]),
        ], axis=1)
    


def _ensure_4d(x: np.ndarray) -> np.ndarray:
    """Akceptuje [N,H,W] lub [N,1,H,W] i zwraca [N,1,H,W]."""
    if x.ndim == 3:
        x = x[:, None, ...]      # [N,H,W] -> [N,1,H,W]
    if x.ndim != 4:
        raise ValueError(f"Expected [N,1,H,W] or [N,H,W], got {x.shape}")
    return x

class Log1pStandardScaler:
    def __init__(self, per_channel=False, eps=1e-12, dtype=np.float64,
                 robust=False, min_std=0.5, winsorize=None):
        """
        robust=True  -> używa mediany i IQR/1.349 zamiast (μ, σ)
        min_std      -> dolne ograniczenie na σ (np. 0.5)
        winsorize    -> tuple (low_p, high_p) np. (0.1, 99.9) do wyznaczania statystyk
        """
        self.per_channel = per_channel
        self.eps = eps
        self.dtype = dtype
        self.robust = robust
        self.min_std = float(min_std)
        self.winsorize = winsorize
        self.fitted_ = False
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X: np.ndarray):
        X = _ensure_4d(X).astype(self.dtype, copy=False)
        Y = np.log1p(X)

        if self.winsorize is not None:
            lo, hi = self.winsorize
            if self.per_channel:
                q_lo = np.percentile(Y, lo, axis=(0,2,3), keepdims=True)
                q_hi = np.percentile(Y, hi, axis=(0,2,3), keepdims=True)
            else:
                q_lo = np.array([[[[np.percentile(Y, lo)]]]], dtype=self.dtype)
                q_hi = np.array([[[[np.percentile(Y, hi)]]]], dtype=self.dtype)
            Y_stats = np.clip(Y, q_lo, q_hi)
        else:
            Y_stats = Y

        if self.robust:
            if self.per_channel:
                med = np.median(Y_stats, axis=(0,2,3), keepdims=True)
                q1  = np.percentile(Y_stats, 25, axis=(0,2,3), keepdims=True)
                q3  = np.percentile(Y_stats, 75, axis=(0,2,3), keepdims=True)
            else:
                med = np.array([[[[np.median(Y_stats)]]]], dtype=self.dtype)
                q1  = np.array([[[[np.percentile(Y_stats, 25)]]]], dtype=self.dtype)
                q3  = np.array([[[[np.percentile(Y_stats, 75)]]]], dtype=self.dtype)
            iqr = np.maximum(q3 - q1, self.eps)
            sigma = (iqr / 1.349)
            mu = med
        else:
            if self.per_channel:
                mu  = Y_stats.mean(axis=(0,2,3), keepdims=True)
                var = Y_stats.var(axis=(0,2,3), keepdims=True)
            else:
                mu  = np.array([[[[Y_stats.mean()]]]], dtype=self.dtype)
                var = np.array([[[[Y_stats.var()]]]], dtype=self.dtype)
            sigma = np.sqrt(np.maximum(var, self.eps))

        sigma = np.maximum(sigma, self.min_std)

        self.mu_ = mu.astype(self.dtype, copy=False)
        self.sigma_ = sigma.astype(self.dtype, copy=False)
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Zastosuj: z = (log1p(X) - mu) / sigma."""
        self._check_fitted()
        X = _ensure_4d(X).astype(self.dtype, copy=False)
        Y = np.log1p(X)
        Z = (Y - self.mu_) / self.sigma_
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit + transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Odwrót: X = expm1(Z*sigma + mu)."""
        self._check_fitted()
        Z = _ensure_4d(Z).astype(self.dtype, copy=False)
        Y = Z * self.sigma_ + self.mu_
        X = np.expm1(Y)
        return X

    def forward_logdet(self, X: np.ndarray) -> np.ndarray:
        """
        Log|det| Jacobianu całego preprocessing’u x->z (pozycja na próbkę):
          y = log1p(x)         : dy/dx = 1/(1+x)   => log|det|_1 = -∑ log(1+x)
          z = (y - mu)/sigma   : dz/dy = 1/sigma   => log|det|_2 = -D * log(sigma)
        Zwraca wektor [N].

        Uwaga: to przydaje się tylko do przeliczenia LL/bpd w oryginalnej domenie x.
        Do trenowania flowa (na z) nie jest potrzebne.
        """
        self._check_fitted()
        X = _ensure_4d(X).astype(self.dtype, copy=False)
        # term1: -sum log(1+x) po wszystkich pikselach
        term1 = -np.log1p(X).sum(axis=(1, 2, 3))  # [N]

        # term2: -D * log(sigma) (D = liczba elementów per próbka)
        N, C, H, W = X.shape
        if self.per_channel:
            # dla każdego kanału: -(H*W)*log(sigma_c), sumujemy po kanałach
            term2_scalar = -(H * W) * np.log(self.sigma_).sum()
            term2 = np.full((N,), term2_scalar, dtype=self.dtype)
        else:
            D = C * H * W
            term2_scalar = - D * np.log(self.sigma_).item()
            term2 = np.full((N,), term2_scalar, dtype=self.dtype)

        return term1 + term2

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Scaler is not fitted yet. Call fit() first.")



def load(
        path,
        data_name,
        scaler_name,
        formatters=None,
        val_size=0.1, test_size=0.2,
        subset_percentage=None
):
    data = np.load(os.path.join(path, f'{data_name}.npz'))['arr_0'].astype(float)

    if subset_percentage is not None:
        slice_index = int(data.shape[0] * subset_percentage)
        data = data[:slice_index]

    data_train, data_test = train_test_split(data, test_size=test_size, random_state=42)
    data_train, data_val = train_test_split(data_train, test_size=val_size / (1 - test_size), random_state=43)

    if scaler_name == 'standard':
        scaler = StandardScaler()
        scaler.fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'log+' in scaler_name:
        degree = int(scaler_name[0])
        eps = float(scaler_name.split('+')[1])
        scaler = LogPlusEpsScaler(degree=degree, eps=eps)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'log1pStd' in scaler_name:
        scaler = Log1pStandardScaler().fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'caloinn' == scaler_name:
        scaler = CaloINNPreprocessorNP().fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'caloinn[z]' == scaler_name:
        scaler = CaloINNPreprocessorNP(zscore=True).fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'caloinnsimp' == scaler_name:
        scaler = CaloINNPreprocessorNPSimple().fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif 'caloinnparticles' == scaler_name:
        scaler = CaloINNParticlesScaler().fit(data)
        data_train, data_val, data_test = map(scaler.transform, (data_train, data_val, data_test))
    elif scaler_name != 'none':
        raise ValueError(f'Unknown Scaler {scaler_name}')
    else:
        scaler = None
    
    if formatters is not None:
        for f in formatters:
            data_train = f(data_train)
            data_val = f(data_val)
            data_test = f(data_test)

    return {
                'train': data_train,
                'val':   data_val,
                'test':  data_test,

                'scaler': scaler,
            }


def batches(*x, batch_size, shuffle_key=None):
    n = len(x[0])

    if shuffle_key is not None:
        perm = jax.random.permutation(shuffle_key, jnp.arange(n))
        x = tuple(x_i[perm] for x_i in x)

    for i in range(0, n, batch_size):
        yield tuple(x_i[i:i + batch_size] for x_i in x)



def load_data_scope(
        data_dir,
        data_load_config,
        train_batch_size,
        eval_batch_size,
        test_batch_size,
        samples_no=None,
        generator=torch.manual_seed(42),
        device='cpu',
        subset_percentage=None,
        **kwargs,
        ):
    
    data_device = device
    train_list = []
    val_list = []
    test_list = []
    scalers = []
    data_names = []
    for data_name, scaler_name, *formatter in data_load_config:
        data_names.append(data_name)
        data_scope = load(
            data_dir,
            data_name=data_name,
            scaler_name=scaler_name,
            formatters=formatter,
            subset_percentage=subset_percentage
        )
        train_list.append(data_scope['train'])
        val_list.append(data_scope['val'])
        test_list.append(data_scope['test'])
        scalers.append(data_scope['scaler'])

    train_dataloader = get_data_loader(train_list, train_batch_size, generator, data_device, shuffle=True)
    val_dataloader   = get_data_loader(val_list,   eval_batch_size,  generator, data_device, shuffle=True)
    test_dataloader  = get_data_loader(test_list,  test_batch_size,  generator, data_device, shuffle=False)

    print(f"Loaded data from {data_dir}")
    return {
        'dataloaders': {
            'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader,
        },
        'dataslices': {
            'train': train_list,
            'val': val_list,
            'test': test_list,
        },
        'n_examples': {
            'train': train_list[0].shape[0],
            'val': val_list[0].shape[0],
            'test': test_list[0].shape[0],
        },
        'samples': [data[:samples_no] for data in train_list],
        'scalers': scalers,
        'data_names': data_names,
    }