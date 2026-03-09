module Main where

import System.Environment (getArgs)
import PlotEmbedding
import TimeSeriesEmbedding

main :: IO ()
main = do
    args  <- getArgs

    case args of
        (tauStr:_) -> do
            let tau = read tauStr :: Integer
            let dataSet = [1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3]
            let normData = normalizeSeries dataSet
            let embedding = returnLag tau normData

            plotTimeSeries embedding "embedding.svg"
            print embedding
        [] -> putStrLn "provide value for tau"
