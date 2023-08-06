# Used car evaluation

This machine-learning model attempts to utilize information on used cars and takes the following attributes into account while rating the vehicle.
1) buying(price):   vhigh, high, med, low.
2) maint(maintenance price):    vhigh, high, med, low.
3) doors:    2, 3, 4, 5more.
4) persons:  2, 4, more.
5) lug_boot: small, med, big.
6) safety:   low, med, high.
7) rating: unacc, acc, good, vgood

Dataset source: https://archive.ics.uci.edu/dataset/19/car+evaluation

These attributes are all categorical and were purposely chosen to make them compatible with decision trees.
Libraries used: pandas, numpy, matplotlib.pyplot, sklearn.tree, sklearn.model_selection, sklearn.metrics

The first step here was to read the data using the pd.read_csv function and then proceeding to assign column names to all the attributes as follows: ["buying", "maint", "doors", "persons", "lug_boot", "safety", "rating"].

Next I split the dataframe into x and y by dropping the column "rating" from the dataframe and assigned it to x, whereas y becomes the value to predicted i.e. column "rating"

Next step would be to replace the categorical string values with 0s and 1s so the data could be read and processed by the imported libraries.


