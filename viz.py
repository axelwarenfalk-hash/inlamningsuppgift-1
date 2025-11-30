import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import HealthAnalyzer 



class HealthVisualizer:
    '''
    En klass för att plotta olika typer av diagram och grafer

    Attributer:
        units = en tabell för måttenheter till varje kolumn i datasetet
        seies = pandas serien som ska plottas
        name = namnet på kolumnen/serien som ska plottas
        unit = enheter som tillhör kolumnnamnet i datasetet.
    '''

    units = {
    'age': 'years',
    'weight': 'Kg',
    'height': 'cm',
    'systolic_bp': 'mmHg',
    'cholesterol': 'mmol',
    'BMI': 'BMI'
    }

    def __init__(self, series):
        self.series = series
        self.name = series.name
        self.unit = self.units.get(self.name, self.name)
        

    def hist22(self, ax, show_mean=False, **kwargs):
        '''
        Ritar ett histogram av dataserien

        args:
            ax = axeln där histogrammet ritas
            show_mean = Om True ritas en vertikal linje vid medelvärdet.
            **kwargs: Extra argument som skickas vidare till ax.hist(). tex färg
        '''
        ax.hist(self.series, edgecolor='white', label=self.name, **kwargs)
        ax.set_xlabel(self.unit)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {self.name}')
        ax.grid(axis='y', alpha=0.5)

        if show_mean:
            ax.axvline(x=self.series.mean(), color='black', linestyle='--', label=f'Mean: {self.series.mean():.2f}')
            ax.legend()


    def bar22(self, ax, xticks_labels=None , **kwargs):
        '''
        Ritar ett stapeldigram över dataserien

        Args:
            ax = axeln där stapeldigrammet ritas
            xticks_labels = Valfria egna etiketter för x-axeln.
            **kwargs: Extra argument som skickas vidare till ax.bar(). tex färg
        '''
        counts = self.series.value_counts()
        ax.bar(counts.index.astype(str), counts.values, edgecolor='white', **kwargs)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Barchart of {self.name}')
        ax.grid(axis='y', alpha=0.5)
        
        if xticks_labels is not None:
            ax.set_xticks(counts.index)
            ax.set_xticklabels(xticks_labels)


    def scatter22(self, other, ax, linear=False, **kwargs):
        '''
        Ritar ett scatterplot mellan två olika dataseries, self och other.
        Kan även lägga till en linjär regressionslinje om så önskas.

        Args:
            other = dataserie
            aax = axeln där scatterplotten ritas
            linear = Om True läggs en linjär regressionslinje till.
            **kwargs: Extra argument som skickas vidare till ax.scatter(). tex färg
        '''
        ax.scatter(self.series, other.series, **kwargs)
        ax.set_xlabel(self.unit)
        ax.set_ylabel(other.unit)
        ax.set_title(f'Scatterplot of {self.name} and {other.name}')
        ax.grid(axis='y', alpha=0.5)

        if linear:
            intercept, slope, _, _ = HealthAnalyzer.linear_regression(self.series, other.series)
            x_vals = np.array([self.series.min(), self.series.max()])
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color='red', label='Linear regression')
        ax.legend()





