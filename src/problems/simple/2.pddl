(define (problem simple-2)
    (:domain vision)
    (:objects red_gem1 - red
              blue_gem1 - blue
              robot1 robot2 - agent
    )
    (:init
        (= (xloc red_gem1) 2)
        (= (yloc red_gem1) 2)
        (= (xloc blue_gem1) 1)
        (= (yloc blue_gem1) 1)
        (= (walls) 
            (transpose (bit-mat 
                (bit-vec 1 1 1 1 1 1 )
                (bit-vec 1 0 1 1 1 1 )
                (bit-vec 1 0 1 0 0 1 )
                (bit-vec 1 0 1 1 1 1 )
                (bit-vec 1 1 1 1 1 1 )))
        )
        (= (xloc robot1) 2)
        (= (yloc robot1) 4)
        (= (xloc robot2) 4)
        (= (yloc robot2) 3)
    )
    (:goal (has robot2 red_gem1))
)
