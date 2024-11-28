(define (problem simple-1)
    (:domain vision)
    (:objects red_gem1 - red
              robot1 - agent
    )
    (:init
        (= (xloc red_gem1) 2)
        (= (yloc red_gem1) 2)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 1 1 1 )
                (bit-vec 1 0 1 )
                (bit-vec 1 0 1 )
                (bit-vec 1 0 1 )
                (bit-vec 1 1 1)))
        )
        (= (xloc robot1) 2)
        (= (yloc robot1) 4)
    )
    (:goal (has robot1 red_gem1))
)
