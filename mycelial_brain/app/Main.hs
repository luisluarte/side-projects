module Main where

import System.Environment (getArgs)
import Core (phi)

main :: IO ()
main = do
    args <- getArgs

    case args of
    -- pattern matching!
        [tauStr, mStr, seriesStr] -> do
            let tau = read tauStr :: Int
            let m = read mStr :: Int
            let series = read seriesStr :: [Double]

            -- the output is as a comment for gnuplot
            let topology = phi tau m series
            mapM_ (putStrLn . formatVector) topology

        _ -> putStrLn "error"


-- helper function to transform into gnuplot ready input
formatVector :: [Double] -> String
formatVector = unwords . map show
