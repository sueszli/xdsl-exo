from exomlir.rewrites.convert_memref_to_llvm import compute_memref_strides


def test_strides_4_4():
    _, result = compute_memref_strides([4, 4])
    assert result == [4, 1]


def test_strides_4_4_4():
    _, result = compute_memref_strides([4, 4, 4])
    assert result == [16, 4, 1]


def test_strides_1_4_8():
    _, result = compute_memref_strides([1, 4, 8])
    assert result == [32, 8, 1]
