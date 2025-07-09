from botorch.test_functions.synthetic import Hartmann
import torch

from gpcheck.data import create_hartmann_dataset


class TestCreateHartmannDataset:
    """test the create_hartmann_dataset function."""

    def test_basic_functionality(self) -> None:
        """test basic functionality of create_hartmann_dataset."""
        train_X, train_Y, test_X, test_Y, f = create_hartmann_dataset(
            seed=42, dim=3, n_train=15, n_test=8
        )

        # check return types
        assert isinstance(train_X, torch.Tensor)
        assert isinstance(train_Y, torch.Tensor)
        assert isinstance(test_X, torch.Tensor)
        assert isinstance(test_Y, torch.Tensor)
        assert isinstance(f, Hartmann)

        # check shapes
        assert train_X.shape == (15, 3)
        assert train_Y.shape == (15, 1)
        assert test_X.shape == (8, 3)
        assert test_Y.shape == (8, 1)

        # check dtypes
        assert train_X.dtype == torch.float64
        assert train_Y.dtype == torch.float64
        assert test_X.dtype == torch.float64
        assert test_Y.dtype == torch.float64

    def test_reproducibility(self) -> None:
        """test that create_hartmann_dataset is reproducible."""
        # create datasets with same seed
        train_X1, train_Y1, test_X1, test_Y1, f1 = create_hartmann_dataset(
            seed=42, dim=3, n_train=10, n_test=5
        )
        train_X2, train_Y2, test_X2, test_Y2, f2 = create_hartmann_dataset(
            seed=42, dim=3, n_train=10, n_test=5
        )

        # should be identical
        torch.testing.assert_close(train_X1, train_X2)
        torch.testing.assert_close(train_Y1, train_Y2)
        torch.testing.assert_close(test_X1, test_X2)
        torch.testing.assert_close(test_Y1, test_Y2)

        # create datasets with different seeds - should be different
        train_X3, train_Y3, test_X3, test_Y3, f3 = create_hartmann_dataset(
            seed=123, dim=3, n_train=10, n_test=5
        )
        assert not torch.allclose(train_X1, train_X3)
        assert not torch.allclose(train_Y1, train_Y3)

    def test_bounds_and_function_consistency(self) -> None:
        """test that generated data is within expected bounds and function evaluates correctly."""
        train_X, train_Y, test_X, test_Y, f = create_hartmann_dataset(
            seed=42, dim=3, n_train=10, n_test=5
        )

        # hartmann function domain is [0, 1]^d
        assert torch.all(train_X >= 0.0)
        assert torch.all(train_X <= 1.0)
        assert torch.all(test_X >= 0.0)
        assert torch.all(test_X <= 1.0)

        # check that Y values are computed correctly
        expected_train_Y = f(train_X).unsqueeze(-1).double()
        expected_test_Y = f(test_X).unsqueeze(-1).double()

        torch.testing.assert_close(train_Y, expected_train_Y)
        torch.testing.assert_close(test_Y, expected_test_Y)

    def test_different_dimensions(self) -> None:
        """test create_hartmann_dataset with different dimensions."""
        for dim in [3, 4, 6]:
            train_X, train_Y, test_X, test_Y, f = create_hartmann_dataset(
                seed=42, dim=dim, n_train=10, n_test=5
            )

            # check dimensions
            assert train_X.shape[1] == dim
            assert test_X.shape[1] == dim
            assert f.dim == dim
