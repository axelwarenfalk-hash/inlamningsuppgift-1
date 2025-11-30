import numpy as np
from sklearn.linear_model import LinearRegression



class HealthAnalyzer:
    
    @staticmethod
    def ci_bootstrap(series, n_bootstraps=5000, confidence=0.95):
        '''
            räknar ut ett konfidensintervall med hjälp av bootstrap

            args:
                series = dataserie från datasetet
                n_bootstraps = antal bootstraps
                confidence = Önskad konfidensnivå (t.ex. 0.95 för 95%).

            Returns:
                - lower = Nedre gränsen av konfidensintervallet.
                - upper = Övre gränsen av konfidensintervallet.
                - mean = Medelvärdet av den ursprungliga dataserien.
        '''
        len_data = len(series)
        # skapar en tom lista med 5000 rader
        boot_means = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            # lägger till medelvärden i listan från ett random sample. Dubletter är tillåtna
            boot_sample = np.random.choice(series, size=len_data, replace=True)
            boot_means[i] = np.mean(boot_sample)

        # ger oss 0.025 (eftersom vi vill räkna ut medelvärden som ligger inom 95 procentilen tar vi bort 2.5% i nedre och 2.5 i övre, totalt 5%.)
        alpha = (1-confidence) / 2
        # ger oss det lägsta värdet inom konfidensintervallet 95%.
        lower = np.percentile(boot_means, 100 * alpha)
        # ger oss det högsta värdet inom konfidensintervallet 95%.
        upper = np.percentile(boot_means, 100 * (1 - alpha))

        return float(lower), float(upper), float(series.mean())
    
    
    @staticmethod
    def linear_regression(series_x, series_y, pred=None):
        '''
        Utför en enkel linjär regression mellan två variabler.
        
        Args:
            series_x = dataserie från datasetet
            series_y = dataserie från datasetet
            pred =  Om angivet, returneras även modellens prediktion för detta x-värde

        Returns:

            intercept = Skärning med y-axeln (β₀).
            slope = Lutningen i regressionen, alltså hur mycket Y förändras per enhets förändring i X.
            - r2 = Förklaringsgraden, anger hur mycket av variationen i Y som förklaras av modellen.
            - residuals = Modellens residualer.
            - prediction = Predikterat värde om pred angivits.

        '''
        # Gör om pandas serie x till en 2 dimensionell matris i numpy
        x = series_x.values.reshape(-1,1)
        # Gör om padas serie till 1 dimensionell array i numpy
        y = series_y.values
        
        # Skapar en linjär regressionsmodell
        model = LinearRegression()
        # Tränar modellen på datan
        model.fit(x, y)

        # Räknar ut skärning med y axel
        intercept = model.intercept_
        # Räknar ut lutningen
        slope = model.coef_[0]

        # Räknar ut r2
        r2 = model.score(x, y)

        # Beräknar modellens prediktioner
        y_prediction = model.predict(x)
        # Residualer = verkligt värde minus predikterat värde
        residuals = y - y_prediction

        # Om användaren fyller i pred
        if pred is not None:
            # Gör en prediktion för det specifika värdet pred
            prediction = model.predict(np.array([[pred]]))[0]
            return intercept, slope, r2, residuals, prediction

        return intercept, slope, r2, residuals