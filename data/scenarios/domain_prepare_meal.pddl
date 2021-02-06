(define (domain meal)
  (:requirements :typing)
  (:types agent location object)
  (:constants
    table shelf1 shelf2 - location
    disk cup - object
    robot - agent
  )
  (:predicates
    (at ?a - agent ?l - location)
    (on ?x - object ?l - location)
    (free ?a - agent)
    (avoid_human ?a - agent)
    (carry ?a - agent ?x - object)
  )

  (:action move
      :parameters (?a - agent ?l1 ?l2 - location)
      :precondition (at ?a ?l1)
      :effect (and (at ?a ?l2) (not (at ?a ?l1)))
      :undo (:action move
                :parameters (?a - agent ?l2 ?l1 - location)
                :precondition (at ?a ?l2)
                :effect (and (at ?a ?l1) (not (at ?a ?l2)))
            )
  )
  (:action pick
      :parameters (?a - agent ?x - object ?l - location)
      :precondition (and (at ?a ?l) (on ?x ?l) (free ?a)) 
      :effect (and (not (on ?x ?l)) (not (free ?a)) (carry ?a ?x))
      :undo (:action place
                :parameters (?a - agent ?x - object ?l - location)
                :precondition (and (at ?a ?l) (carry ?a ?x)) 
                :effect (and (not (carry ?a ?x)) (on ?x ?l) (free ?a))
            )
  )
  (:action place
      :parameters (?a - agent ?x - object ?l - location)
      :precondition (and (at ?a ?l) (carry ?a ?x)) 
      :effect (and (not (carry ?a ?x)) (on ?x ?l) (free ?a))
      :undo (:action pick
                :parameters (?a - agent ?x - object ?l - location)
                :precondition (and (at ?a ?l) (on ?x ?l) (free ?a)) 
                :effect (and (not (on ?x ?l)) (not (free ?a)) (carry ?a ?x))
            )
  )
)