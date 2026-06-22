use crate::numeric::Compare;

pub(crate) fn where_cond<T>(mask: &T::Mask, on_true: T, on_false: T) -> T
where
    T: Compare,
{
    T::select(mask, on_true, on_false)
}
