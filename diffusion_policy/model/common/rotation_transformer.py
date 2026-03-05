from typing import Union
import torch
import numpy as np
import functools

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
            from_rep='axis_angle',
            to_rep='rotation_6d',
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        # store params for lazy init
        self._from_rep = from_rep
        self._to_rep = to_rep
        self._from_convention = from_convention
        self._to_convention = to_convention
        self._initialized = False
        self.forward_funcs = list()
        self.inverse_funcs = list()

    def _lazy_init(self):
        if self._initialized:
            return
        import pytorch3d.transforms as pt

        forward_funcs = list()
        inverse_funcs = list()

        if self._from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{self._from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{self._from_rep}')
            ]
            if self._from_convention is not None:
                funcs = [functools.partial(func, convention=self._from_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if self._to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{self._to_rep}'),
                getattr(pt, f'{self._to_rep}_to_matrix')
            ]
            if self._to_convention is not None:
                funcs = [functools.partial(func, convention=self._to_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs
        self._initialized = True

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        self._lazy_init()
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        self._lazy_init()
        return self._apply_funcs(x, self.inverse_funcs)
