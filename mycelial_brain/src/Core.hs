module Core where

-- function phi is the delay embedding map
-- takes to Int as parameters
-- takes a vector with the number sequence
-- output a list of list that are the embeddings
phi :: Int -> Int -> [Double] -> [[Double]]
phi tau m series
    | length series <= burnIn = [] -- series must be greater that the burnIn
    -- set builder notation
    -- we are defining the set of the number sequence
    -- generateVector is a function that takes t as argument
    -- t is a subset in validIndices
    | otherwise = [ generateVector t | t <- validIndices ]
    -- here we start the definition of our functions
    where
        -- points consumed before starting the embedding
        burnIn = (m - 1) * tau

        -- the validIndices are from the burnIn up to the end of the series
        validIndices = [burnIn .. length series - 1]

        -- now the vector is generated
        -- !! takes a list and returns the value at the index
        -- the vector (k * tau) performs broadcast substraction in t
        -- for every index in t we get back a vector of the same size as
        --  the resulting from (k * tau), and again for the substraction
        --  at each index of t
        generateVector t = [ series !! (t - k * tau) | k <- [0 .. m - 1] ]

