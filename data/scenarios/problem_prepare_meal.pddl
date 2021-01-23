(define (problem prepare_meal)
    (:domain meal)
    (:init
        (at pr2 table)
        (on disk shelf1)
        (on cup shelf2)
        (free pr2)
        (avoid_human pr2)
    )
    (:goal (and
        (on disk table)
        (on cup table)
    ))
)