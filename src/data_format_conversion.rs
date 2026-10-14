use half::bf16;

pub fn to_bf16<T: Into<f64>>(x: T) -> bf16 {
    bf16::from_f64(x.into())
}

pub fn to_bf16s<T: Into<f64> + Copy>(x: &[T]) -> Vec<bf16> {
    x.iter().map(|&x| to_bf16(x)).collect()
}
