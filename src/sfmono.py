import pandas as pd
import numpy as np
import altair as alt

def sfmono():
    font = "SFMono-Regular"
    
    return {
        "config" : {
             "title": {
                 'font': font, 
                 'fontSize': 13
             },
             "axis": {
                  "titleFont": font,
                  "titleFontSize": 12,
                  "labelFont": font,
                  "labelFontSize": 11
             },
            "view": {
                'width': 500, 
                'height': 400
            },
            "legend": {
                "titleFont": font,      
                "titleFontSize": 12,
                "labelFont": font,
                "labelFontSize": 11
            },
            "text": {
                "font": font,
                "fontSize": 12,
            },
            "header": {
                "titleFont": font,
                "titleFontSize": 11,
                "labelFont": font,
                "labelFontSize": 11
            }
        } # config
    } # return

alt.themes.register('sfmono', sfmono)
alt.themes.enable('sfmono')


