#[test]
fn factor() {
    accessor_agrees!(crate::Factor, as_f64);
    sign_predicates_agree!(crate::Factor, as_f64);
    finiteness_agrees!(crate::Factor);
    display_distinguishes!(crate::Factor);
    epsilon_predicate_agrees!(crate::Factor);
}

#[test]
fn percentage() {
    accessor_agrees!(crate::Percentage, as_percent);
    sign_predicates_agree!(crate::Percentage, as_percent);
    display_distinguishes!(crate::Percentage);
}
