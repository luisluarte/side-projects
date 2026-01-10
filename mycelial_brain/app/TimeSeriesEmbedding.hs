module TimeSeriesEmbedding where

returnLag :: Integer -> [Double] -> [(Double, Double)]
returnLag _ [] = []
returnLag tau (x:xs)
    | length xs < fromIntegral tau = []
    | length xs == fromIntegral tau = [(x, xs !! (fromIntegral tau - 1))]
    | otherwise = (x, xs !! (fromIntegral tau - 1)) : returnLag tau xs

normalizeSeries :: [Double] -> [Double]
normalizeSeries [] = []
normalizeSeries xs
    | maxVal == minVal = replicate (length xs) 0.0
    | otherwise = map (\x -> (x - minVal) / range) xs
    where
        minVal = minimum xs
        maxVal = maximum xs
        range = maxVal - minVal
