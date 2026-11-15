//! The sugar: `property!` to declare a test, `check!` family to assert inside one.
//!
//! # Why assertions are not `assert!`
//!
//! A property body returns [`Outcome`](crate::Outcome) rather than panicking, because the
//! runner re-runs it hundreds of times while shrinking and a `Result` is cheaper than an unwind
//! per probe. `check!` and friends therefore `return Err(...)` instead of panicking — which is
//! why they only work inside a `property!` body, and why they are named differently from
//! `assert!` rather than shadowing it. A plain `assert!` still works (the runner catches the
//! unwind) but costs more and reports less.

/// Declare one or more properties as `#[test]` functions.
///
/// ```
/// use euclid_property::{check_eq, generate, property};
///
/// property! {
///     /// Doc comments carry through to the generated test.
///     fn addition_commutes(a in generate::int(-100, 100), b in generate::int(-100, 100)) {
///         check_eq!(a + b, b + a);
///     }
/// }
/// ```
///
/// Each declaration takes bindings of the form `name in generator`, and the body is the
/// property. An optional leading `#![config(expr)]` sets the [`Config`](crate::Config) for
/// every property in the block.
///
/// # Do not write `#[test]` yourself
///
/// This macro supplies it. Writing it as well is accepted by the compiler and makes libtest
/// register the function *twice*, so every property silently runs two of itself — which looks
/// like a passing suite with a suspiciously round test count. It is the one mistake this macro
/// invites, and it is what happens when a suite is ported over from a harness whose macro did
/// not add the attribute. Attributes other than `#[test]` pass through as written.
#[macro_export]
macro_rules! property {
    // With a block-level config.
    (
        #![config($config:expr)]
        $(
            $(#[$meta:meta])*
            fn $name:ident($($binding:pat_param in $generator:expr),* $(,)?) $body:block
        )*
    ) => {
        $(
            $(#[$meta])*
            #[test]
            fn $name() {
                let config = $config;
                let generator = $crate::__tuple_gen!($($generator),*);
                $crate::check(
                    stringify!($name),
                    &config,
                    &generator,
                    |$crate::__tuple_pat!($($binding),*)| {
                        $body
                        Ok(())
                    },
                );
            }
        )*
    };
    // Without one.
    (
        $(
            $(#[$meta:meta])*
            fn $name:ident($($binding:pat_param in $generator:expr),* $(,)?) $body:block
        )*
    ) => {
        $crate::property! {
            #![config($crate::Config::default())]
            $(
                $(#[$meta])*
                fn $name($($binding in $generator),*) $body
            )*
        }
    };
}

/// Combine 1–4 generators into one, matching `__tuple_pat`.
#[doc(hidden)]
#[macro_export]
macro_rules! __tuple_gen {
    ($a:expr) => {
        $a
    };
    ($a:expr, $b:expr) => {
        $crate::generate::pair($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
        $crate::generate::triple($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        $crate::generate::pair(
            $crate::generate::pair($a, $b),
            $crate::generate::pair($c, $d),
        )
    };
}

/// Destructure what `__tuple_gen` produced.
#[doc(hidden)]
#[macro_export]
macro_rules! __tuple_pat {
    ($a:pat) => {
        $a
    };
    ($a:pat, $b:pat) => {
        ($a, $b)
    };
    ($a:pat, $b:pat, $c:pat) => {
        ($a, $b, $c)
    };
    ($a:pat, $b:pat, $c:pat, $d:pat) => {
        (($a, $b), ($c, $d))
    };
}

/// Fail the property unless `condition` holds.
#[macro_export]
macro_rules! check {
    ($condition:expr) => {
        $crate::check!($condition, concat!("check failed: ", stringify!($condition)))
    };
    ($condition:expr, $($arg:tt)+) => {
        if !$condition {
            return Err($crate::Failure::new(
                format!($($arg)+),
                concat!(file!(), ":", line!()),
            ));
        }
    };
}

/// Fail the property unless `left == right`, reporting both.
#[macro_export]
macro_rules! check_eq {
    ($left:expr, $right:expr) => {
        $crate::check_eq!($left, $right, "")
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let left = $left;
        let right = $right;
        if left != right {
            let context = format!($($arg)+);
            let separator = if context.is_empty() { "" } else { ": " };
            return Err($crate::Failure::new(
                format!(
                    "{}{}{} != {}\n    left:  {:?}\n    right: {:?}",
                    context,
                    separator,
                    stringify!($left),
                    stringify!($right),
                    left,
                    right,
                ),
                concat!(file!(), ":", line!()),
            ));
        }
    }};
}

/// Fail the property unless `left != right`.
#[macro_export]
macro_rules! check_ne {
    ($left:expr, $right:expr) => {
        $crate::check_ne!($left, $right, "")
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let left = $left;
        let right = $right;
        if left == right {
            let context = format!($($arg)+);
            let separator = if context.is_empty() { "" } else { ": " };
            return Err($crate::Failure::new(
                format!(
                    "{}{}{} == {} (expected them to differ)\n    both: {:?}",
                    context,
                    separator,
                    stringify!($left),
                    stringify!($right),
                    left,
                ),
                concat!(file!(), ":", line!()),
            ));
        }
    }};
}

/// Skip this case unless `condition` holds.
///
/// Prefer narrowing the generator. An `assume!` discards work already done and counts against
/// the rejection budget, so a property that assumes away most of its input is quietly testing
/// much less than its case count suggests — which is exactly why the runner reports the
/// rejection rate instead of letting it pass unnoticed.
#[macro_export]
macro_rules! assume {
    ($condition:expr) => {
        if !$condition {
            return Ok(());
        }
    };
}
