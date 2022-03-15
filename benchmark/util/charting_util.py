# -*- coding: utf-8 -*-
"""
Module with utilities to help with drawing charts.
"""

import math
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd 
import pandas.api 

A4_LONG_SIDE = 11.69
A4_SHORT_SIDE = 8.27

def save_chart(
        df: pd.DataFrame,
        title: str,
        output_path: Path,
        format_y_as_pct: bool = False,
        print_labels_on_points: bool = False,
        open_output_file: bool = False,
        size: Tuple[float, float] = (8, 4),
        plot_kind: str = "line",
        linestyle: Optional[str] = None):
    
    ### Init ###
    # Check input
    non_numeric_columns = [
            column for column in df.columns if not pandas.api.types.is_numeric_dtype(df[column])]
    if len(non_numeric_columns) > 0:
        raise Exception(f"df has non-numeric columns, so cannot be plotted: {non_numeric_columns}")
    
    # Init some things based on input
    rot = 90

    ### Prepare plot figure and axes ###
    fig, axs = plt.subplots(figsize=(size))    
    # Make sure all x axis values are shown
    axs.set_xticks(range(len(df)))
    # Show y axes as percentages is asked
    if format_y_as_pct is True:
        axs.yaxis.set_major_formatter(plt.FuncFormatter("{0:.2%}".format))

    ### Plot ###
    df.plot(ax=axs, kind=plot_kind, rot=rot, title=title, linestyle=linestyle)
    
    ### Set different markers + print labels ###
    # Set different markers for each line + get mn/max values + print labels
    markers = ("+", ".", "o", "*")
    max_y_value = None
    min_y_value = None
    for i, line in enumerate(axs.get_lines()):
        line.set_marker(markers[i%len(markers)])
        for row in df.itertuples():
            for row_fieldname, row_fieldvalue in row._asdict().items():
                if row_fieldname != "Index":
                    if max_y_value is None or row_fieldvalue > max_y_value:
                        max_y_value = row_fieldvalue
                    if min_y_value is None or row_fieldvalue < min_y_value:
                        min_y_value = row_fieldvalue
                    if print_labels_on_points is True:
                        text = f"{row_fieldvalue*100:.2f}"
                        axs.annotate(
                                text=text, # type: ignore
                                #s=text,
                                xy=(row.Index, row_fieldvalue),
                                xytext=(0, 5), 
                                textcoords="offset points",
                                ha='center')
    
    # Set bottom and top values for y axis
    if max_y_value is not None:
        max_y_value *= 1.1
    if max_y_value is not None and math.isnan(max_y_value) is False:
        plt.ylim(bottom=0, top=max_y_value)
    else:
        plt.ylim(bottom=0)

    ### Set legend to the right of the chart ###
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    ### Save and open if wanted ###
    fig.savefig(output_path)
    if open_output_file is True:
        os.startfile(output_path)
