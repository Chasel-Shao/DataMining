# Examples and Exercises from Think Stats, 2nd Edition

http://thinkstats2.com

Copyright 2016 Allen B. Downey

MIT License: https://opensource.org/licenses/MIT

```python
from __future__ import print_function, division

import nsfg
```

## Examples from Chapter 1

Read NSFG data into a Pandas DataFrame.

```python
preg = nsfg.ReadFemPreg()
preg.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caseid</th>
      <th>pregordr</th>
      <th>howpreg_n</th>
      <th>howpreg_p</th>
      <th>moscurrp</th>
      <th>nowprgdk</th>
      <th>pregend1</th>
      <th>pregend2</th>
      <th>nbrnaliv</th>
      <th>multbrth</th>
      <th>...</th>
      <th>laborfor_i</th>
      <th>religion_i</th>
      <th>metro_i</th>
      <th>basewgt</th>
      <th>adj_mod_basewgt</th>
      <th>finalwgt</th>
      <th>secu_p</th>
      <th>sest</th>
      <th>cmintvw</th>
      <th>totalwgt_lb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3410.389399</td>
      <td>3869.349602</td>
      <td>6448.271112</td>
      <td>2</td>
      <td>9</td>
      <td>NaN</td>
      <td>8.8125</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3410.389399</td>
      <td>3869.349602</td>
      <td>6448.271112</td>
      <td>2</td>
      <td>9</td>
      <td>NaN</td>
      <td>7.8750</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7226.301740</td>
      <td>8567.549110</td>
      <td>12999.542264</td>
      <td>2</td>
      <td>12</td>
      <td>NaN</td>
      <td>9.1250</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7226.301740</td>
      <td>8567.549110</td>
      <td>12999.542264</td>
      <td>2</td>
      <td>12</td>
      <td>NaN</td>
      <td>7.0000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7226.301740</td>
      <td>8567.549110</td>
      <td>12999.542264</td>
      <td>2</td>
      <td>12</td>
      <td>NaN</td>
      <td>6.1875</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 244 columns</p>


Print the column names.


```python
preg.columns
```


    Index(['caseid', 'pregordr', 'howpreg_n', 'howpreg_p', 'moscurrp', 'nowprgdk',
           'pregend1', 'pregend2', 'nbrnaliv', 'multbrth',
           ...
           'laborfor_i', 'religion_i', 'metro_i', 'basewgt', 'adj_mod_basewgt',
           'finalwgt', 'secu_p', 'sest', 'cmintvw', 'totalwgt_lb'],
          dtype='object', length=244)

Select a single column name.


```python
preg.columns[1]
```


    'pregordr'

Select a column and check what type it is.


```python
pregordr = preg['pregordr']
type(pregordr)
```


    pandas.core.series.Series

Print a column.


```python
pregordr
```


    0        1
    1        2
    2        1
    3        2
    4        3
            ..
    13588    1
    13589    2
    13590    3
    13591    4
    13592    5
    Name: pregordr, Length: 13593, dtype: int64

Select a single element from a column.


```python
pregordr[0]
```
    1


Select a slice from a column.


```python
pregordr[2:5]
```

    2    1
    3    2
    4    3
    Name: pregordr, dtype: int64


Select a column using dot notation.


```python
pregordr = preg.pregordr
```

Count the number of times each value occurs.


```python
preg.outcome.value_counts().sort_index()
```

    1    9148
    2    1862
    3     120
    4    1921
    5     190
    6     352
    Name: outcome, dtype: int64


Check the values of another variable.


```python
preg.birthwgt_lb.value_counts().sort_index()
```


    0.0        8
    1.0       40
    2.0       53
    3.0       98
    4.0      229
    5.0      697
    6.0     2223
    7.0     3049
    8.0     1889
    9.0      623
    10.0     132
    11.0      26
    12.0      10
    13.0       3
    14.0       3
    15.0       1
    Name: birthwgt_lb, dtype: int64

Make a dictionary that maps from each respondent's `caseid` to a list of indices into the pregnancy `DataFrame`.  Use it to select the pregnancy outcomes for a single respondent.


```python
caseid = 10229
preg_map = nsfg.MakePregMap(preg)
indices = preg_map[caseid]
preg.outcome[indices].values
```


    array([4, 4, 4, 4, 4, 4, 1])

## Exercises

Select the `birthord` column, print the value counts, and compare to results published in the [codebook](http://www.icpsr.umich.edu/nsfg6/Controller?displayPage=labelDetails&fileCode=PREG&section=A&subSec=8016&srtLabel=611933)


```python
# Solution goes here
# Select a column using dot notation or use as a dictionary
# birthord = preg['birthord']
preg.birthord.value_counts().sort_index()
```


    1.0     4413
    2.0     2874
    3.0     1234
    4.0      421
    5.0      126
    6.0       50
    7.0       20
    8.0        7
    9.0        2
    10.0       1
    Name: birthord, dtype: int64

We can also use `isnull` to count the number of nans.


```python
preg.birthord.isnull().sum()
```


    4445

Select the `prglngth` column, print the value counts, and compare to results published in the [codebook](http://www.icpsr.umich.edu/nsfg6/Controller?displayPage=labelDetails&fileCode=PREG&section=A&subSec=8016&srtLabel=611931)


```python
# Solution goes here
prglngth = preg.prglngth.value_counts().sort_index()
print("0-13: " + str(prglngth[0:14].sum()))
print("14-27: " + str(prglngth[14:27].sum()))
print("27-50: " + str(prglngth[27:50].sum()))
```

    0-13: 3522
    14-27: 793
    27-50: 9278


To compute the mean of a column, you can invoke the `mean` method on a Series.  For example, here is the mean birthweight in pounds:


```python
preg.totalwgt_lb.mean()
```


    7.265628457623368

Create a new column named <tt>totalwgt_kg</tt> that contains birth weight in kilograms.  Compute its mean.  Remember that when you create a new column, you have to use dictionary syntax, not dot notation.


```python
# Solution goes here
totalwgt_kg = preg.totalwgt_lb / 2.2046
preg['totalwgt_kg'] = totalwgt_kg
preg['totalwgt_kg'].mean()
```


    3.29566744879946

`nsfg.py` also provides `ReadFemResp`, which reads the female respondents file and returns a `DataFrame`:


```python
resp = nsfg.ReadFemResp()
```

`DataFrame` provides a method `head` that displays the first five rows:


```python
resp.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caseid</th>
      <th>rscrinf</th>
      <th>rdormres</th>
      <th>rostscrn</th>
      <th>rscreenhisp</th>
      <th>rscreenrace</th>
      <th>age_a</th>
      <th>age_r</th>
      <th>cmbirth</th>
      <th>agescrn</th>
      <th>...</th>
      <th>pubassis_i</th>
      <th>basewgt</th>
      <th>adj_mod_basewgt</th>
      <th>finalwgt</th>
      <th>secu_r</th>
      <th>sest</th>
      <th>cmintvw</th>
      <th>cmlstyr</th>
      <th>screentime</th>
      <th>intvlngth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2298</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>5.0</td>
      <td>27</td>
      <td>27</td>
      <td>902</td>
      <td>27</td>
      <td>...</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>1234</td>
      <td>1222</td>
      <td>18:26:36</td>
      <td>110.492667</td>
    </tr>
    <tr>
      <td>1</td>
      <td>5012</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>42</td>
      <td>42</td>
      <td>718</td>
      <td>42</td>
      <td>...</td>
      <td>0</td>
      <td>2335.279149</td>
      <td>2846.799490</td>
      <td>4744.191350</td>
      <td>2</td>
      <td>18</td>
      <td>1233</td>
      <td>1221</td>
      <td>16:30:59</td>
      <td>64.294000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>11586</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5.0</td>
      <td>43</td>
      <td>43</td>
      <td>708</td>
      <td>43</td>
      <td>...</td>
      <td>0</td>
      <td>2335.279149</td>
      <td>2846.799490</td>
      <td>4744.191350</td>
      <td>2</td>
      <td>18</td>
      <td>1234</td>
      <td>1222</td>
      <td>18:19:09</td>
      <td>75.149167</td>
    </tr>
    <tr>
      <td>3</td>
      <td>6794</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>15</td>
      <td>15</td>
      <td>1042</td>
      <td>15</td>
      <td>...</td>
      <td>0</td>
      <td>3783.152221</td>
      <td>5071.464231</td>
      <td>5923.977368</td>
      <td>2</td>
      <td>18</td>
      <td>1234</td>
      <td>1222</td>
      <td>15:54:43</td>
      <td>28.642833</td>
    </tr>
    <tr>
      <td>4</td>
      <td>616</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>5.0</td>
      <td>20</td>
      <td>20</td>
      <td>991</td>
      <td>20</td>
      <td>...</td>
      <td>0</td>
      <td>5341.329968</td>
      <td>6437.335772</td>
      <td>7229.128072</td>
      <td>2</td>
      <td>18</td>
      <td>1233</td>
      <td>1221</td>
      <td>14:19:44</td>
      <td>69.502667</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3087 columns</p>


Select the `age_r` column from `resp` and print the value counts.  How old are the youngest and oldest respondents?


```python
# Solution goes here
resp['age_r'].value_counts().sort_index()
```


    15    217
    16    223
    17    234
    18    235
    19    241
    20    258
    21    267
    22    287
    23    282
    24    269
    25    267
    26    260
    27    255
    28    252
    29    262
    30    292
    31    278
    32    273
    33    257
    34    255
    35    262
    36    266
    37    271
    38    256
    39    215
    40    256
    41    250
    42    215
    43    253
    44    235
    Name: age_r, dtype: int64

We can use the `caseid` to match up rows from `resp` and `preg`.  For example, we can select the row from `resp` for `caseid` 2298 like this:


```python
resp[resp.caseid==2298]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caseid</th>
      <th>rscrinf</th>
      <th>rdormres</th>
      <th>rostscrn</th>
      <th>rscreenhisp</th>
      <th>rscreenrace</th>
      <th>age_a</th>
      <th>age_r</th>
      <th>cmbirth</th>
      <th>agescrn</th>
      <th>...</th>
      <th>pubassis_i</th>
      <th>basewgt</th>
      <th>adj_mod_basewgt</th>
      <th>finalwgt</th>
      <th>secu_r</th>
      <th>sest</th>
      <th>cmintvw</th>
      <th>cmlstyr</th>
      <th>screentime</th>
      <th>intvlngth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2298</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>5.0</td>
      <td>27</td>
      <td>27</td>
      <td>902</td>
      <td>27</td>
      <td>...</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>1234</td>
      <td>1222</td>
      <td>18:26:36</td>
      <td>110.492667</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 3087 columns</p>


And we can get the corresponding rows from `preg` like this:


```python
preg[preg.caseid==2298]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caseid</th>
      <th>pregordr</th>
      <th>howpreg_n</th>
      <th>howpreg_p</th>
      <th>moscurrp</th>
      <th>nowprgdk</th>
      <th>pregend1</th>
      <th>pregend2</th>
      <th>nbrnaliv</th>
      <th>multbrth</th>
      <th>...</th>
      <th>laborfor_i</th>
      <th>religion_i</th>
      <th>metro_i</th>
      <th>basewgt</th>
      <th>adj_mod_basewgt</th>
      <th>finalwgt</th>
      <th>secu_p</th>
      <th>sest</th>
      <th>cmintvw</th>
      <th>totalwgt_lb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2610</th>
      <td>2298</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>6.8750</td>
    </tr>
    <tr>
      <th>2611</th>
      <td>2298</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>5.5000</td>
    </tr>
    <tr>
      <th>2612</th>
      <td>2298</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>4.1875</td>
    </tr>
    <tr>
      <th>2613</th>
      <td>2298</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>6.8750</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 244 columns</p>

How old is the respondent with `caseid` 1?

```python
# Solution goes here
resp[resp.caseid==1].age_r
```

    1069    44
    Name: age_r, dtype: int64

What are the pregnancy lengths for the respondent with `caseid` 2298?


```python
# Solution goes here
preg[preg.caseid==2298].prglngth
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>caseid</th>
      <th>pregordr</th>
      <th>howpreg_n</th>
      <th>howpreg_p</th>
      <th>moscurrp</th>
      <th>nowprgdk</th>
      <th>pregend1</th>
      <th>pregend2</th>
      <th>nbrnaliv</th>
      <th>multbrth</th>
      <th>...</th>
      <th>religion_i</th>
      <th>metro_i</th>
      <th>basewgt</th>
      <th>adj_mod_basewgt</th>
      <th>finalwgt</th>
      <th>secu_p</th>
      <th>sest</th>
      <th>cmintvw</th>
      <th>totalwgt_lb</th>
      <th>totalwgt_kg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2610</td>
      <td>2298</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>6.8750</td>
      <td>3.118480</td>
    </tr>
    <tr>
      <td>2611</td>
      <td>2298</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>5.5000</td>
      <td>2.494784</td>
    </tr>
    <tr>
      <td>2612</td>
      <td>2298</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>4.1875</td>
      <td>1.899438</td>
    </tr>
    <tr>
      <td>2613</td>
      <td>2298</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3247.916977</td>
      <td>5123.759559</td>
      <td>5556.717241</td>
      <td>2</td>
      <td>18</td>
      <td>NaN</td>
      <td>6.8750</td>
      <td>3.118480</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 245 columns</p>
What was the birthweight of the first baby born to the respondent with `caseid` 5012?


```python
# Solution goes here
preg[preg.caseid==5012].birthwgt_lb
```


    5515    6.0
    Name: birthwgt_lb, dtype: float64
