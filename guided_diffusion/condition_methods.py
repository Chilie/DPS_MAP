from abc import ABC, abstractmethod
import torch
import numpy as np

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t

@register_conditioning_method(name='dmps')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        # alpha_bar = np.clip(alpha_bar, 1e-16, 1-1e-16)
        # alpha_t = np.clip(alpha_t, 1e-16, 1-1e-16)
        scale_S = (1-alpha_bar) #/(alpha_bar) alpha_bar*
        

        S_vector = (1/(S*scale_S + alpha_bar*noise_std**2)).to(x_t.device).reshape(-1,1)
        # Temp_value = H_funcs.Ut(np.sqrt(alpha_bar)*measurement - H_funcs.H(x_t)).t()
        # print(S_vector.shape)
        # print(Temp_value.t().shape)
        # grad_value = H_funcs.Ht(H_funcs.U((S_vector*Temp_value).t()))
        Temp_value = H_funcs.Ut(np.sqrt(alpha_bar)*measurement - H_funcs.H(x_t)) #.t()
        # print(S_vector.t().shape)
        # print(measurement.shape)
        # print(Temp_value.shape)
        grad_value = H_funcs.Ht(H_funcs.U((S_vector.t()*Temp_value))) #.t()))
 
        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)
        # x_t += self.scale*grad_value *(1-alpha_t)/np.sqrt(alpha_t)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value #*(1-alpha_t)/np.sqrt(alpha_t)
        return x_t

@register_conditioning_method(name='dmps_ours')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        # alpha_bar = np.clip(alpha_bar, 1e-16, 1-1e-16)
        # alpha_t = np.clip(alpha_t, 1e-16, 1-1e-16)
        scale_S = (1-alpha_bar) #/(alpha_bar)
        
        idx = kwargs.get('idx') #alpha_bar*

        S_vector = (1/(S*scale_S + alpha_bar*noise_std**2)).to(x_t.device).reshape(-1,1) #alpha_bar*
        
        if idx >=0:
            Temp_value = H_funcs.Ut(np.sqrt(alpha_bar)*measurement - H_funcs.H(np.sqrt(alpha_bar)*x_0_hat)) #x_t)) #.t()
        else:
            Temp_value = H_funcs.Ut(np.sqrt(alpha_bar)*measurement - H_funcs.H(x_t))

        grad_value = H_funcs.Ht(H_funcs.U((S_vector.t()*Temp_value))) #.t()))
 
        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value 
        return x_t

@register_conditioning_method(name='dmps_song')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        singulars = H_funcs.singulars()
        S = singulars*singulars.to(x_t.device)
        scale_S = (1-alpha_bar) #/(alpha_bar)
        

        S_vector = (1/(S*scale_S + alpha_bar*noise_std**2)).to(x_t.device).reshape(-1,1)
        x_0_hat_s = np.sqrt(alpha_bar) * x_0_hat
        Temp_value = H_funcs.Ut(np.sqrt(alpha_bar)*measurement - H_funcs.H(x_0_hat_s)) #.t()

        grad_value_det = H_funcs.Ht(H_funcs.U((S_vector.t()*Temp_value))).detach() #.t()))

        loss = (grad_value_det*x_0_hat.reshape(x_0_hat.shape[0],-1)).sum()
        grad_value = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value #*(1-alpha_t)/np.sqrt(alpha_t)
        return x_t
import linops as lo
from linops.cg import CG

# import pytest
class LinearOperator_my(lo.LinearOperator):
    r"""
    A LinearOperator representing a diagonal matrix.
    """
    # def __init__(self, A):
    #     # diag: the vector that defines the diagonal of the matrix
    #     super().__init__(A = A)
    #     # self.diag = diag
    #     self.A = A
    def __init__(self, A, y, alpha, sigma, shape):
        self._A = A
        self._y = y
        self.alpha = alpha
        self.sigma = sigma
        self._adjoint = self
        self._shape = shape
        self.supports_operator_matrix = True

    def _matmul_impl(self, v):
        out = self.alpha*self._y*v + self.sigma * self._A.forward(self._A.transpose(v))
        # return torch.matmul(self._A,v) #.T).T
        return out

@register_conditioning_method(name='dmps_poisson')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        
        # D = LinearOperator_my(A = H_funcs, y = (measurement.abs())/10, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        D = LinearOperator_my(A = H_funcs, y = noise_std**2, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        # there are two parts of the grad, we compute them one-by-one
        cg = CG(D,maxiter=20, X0=None,verbose=False) #verbose=True
        x = cg(np.sqrt(alpha_bar)*measurement - H_funcs.forward(x_t))

        grad_value = H_funcs.transpose(x)
 
        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value #*(1-alpha_t)/np.sqrt(alpha_t)
        return x_t
    
@register_conditioning_method(name='dmps_poisson_ours')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        
        # define the CG problem
        # D = LinearOperator_my(A = H_funcs, y = (measurement.abs())/10, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        D = LinearOperator_my(A = H_funcs, y = noise_std**2, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        # there are two parts of the grad, we compute them one-by-one
        cg = CG(D,maxiter=20, X0=None,verbose=False) #verbose=True
        # x = cg(np.sqrt(alpha_bar)*measurement - H_funcs.forward(x_t)) #np.sqrt(alpha_bar)*x_0_hat

        x = cg(np.sqrt(alpha_bar)*measurement - H_funcs.forward(np.sqrt(alpha_bar)*x_0_hat)) 

        grad_value = H_funcs.transpose(x)
 
        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value #*(1-alpha_t)/np.sqrt(alpha_t)
        return x_t

@register_conditioning_method(name='dmps_poisson_song')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, noise_std, alpha_t, alpha_bar, pseudonoise_scale,  **kwargs):
        
        # define the CG problem
        # D = LinearOperator_my(A = H_funcs, y = (measurement.abs())/10, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        D = LinearOperator_my(A = H_funcs, y = noise_std**2, alpha=alpha_bar, sigma=(1-alpha_bar),shape=(measurement.shape[1],measurement.shape[1]))
        # there are two parts of the grad, we compute them one-by-one
        cg = CG(D,maxiter=20, X0=None,verbose=True)
        # x = cg(np.sqrt(alpha_bar)*measurement - H_funcs.forward(x_t)) #np.sqrt(alpha_bar)*x_0_hat

        x = cg(np.sqrt(alpha_bar)*measurement - H_funcs.forward(np.sqrt(alpha_bar)*x_0_hat)) 

        
        grad_value_det = H_funcs.transpose(x)
        loss = (grad_value_det*x_0_hat).sum()
        grad_value = torch.autograd.grad(outputs=loss, inputs=x_prev)[0]

        grad_value = grad_value.reshape(x_t.shape) #/np.sqrt(alpha_bar)

        step = self.scale*kwargs.get('scale_coef') #kwargs.get('out_coef', 1.0)
        x_t += step*grad_value #*(1-alpha_t)/np.sqrt(alpha_t)
        return x_t