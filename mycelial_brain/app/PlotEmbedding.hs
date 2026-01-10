module PlotEmbedding where

import Graphics.Rendering.Chart.Easy
import Graphics.Rendering.Chart.Backend.Diagrams

plotTimeSeries :: [(Double, Double)] -> String -> IO ()
plotTimeSeries values filename = toFile def filename $ do
    layout_title .= "time series embedding (lag plot)"
    layout_x_axis . laxis_generate .= scaledAxis def (-0.1, 1.1)
    layout_y_axis . laxis_generate .= scaledAxis def (-0.1, 1.1)
    layout_x_axis . laxis_title .= "t"
    layout_y_axis . laxis_title .= "t + tau"

    plot (line "trajectory" [values])
    plot (points "embedding" values)
