use crate::numeric::LazyCompare;

pub(crate) fn where_cond<T>(mask: &T::Mask, on_true: T, on_false: T) -> T
where
    T: LazyCompare,
{
    T::select(mask, on_true, on_false)
}
