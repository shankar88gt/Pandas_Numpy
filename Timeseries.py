"""
TimeSeries in Python & Key items

Basic building block 
    Pandas package & datetime package

    timestamp = pd.Timestamp('2017-01-01') - to convert date to timestamp
    timestamp.year - 2017
    timestamp.day_name() - Sunday

    pd. Period & freq
        period = pd.Period('2017-01') - default month end
        period.asfreq('D') - convert to daily
        period.to_timestamp().to_period('M') - convert period to timestamp & back

    Frequency info enables basic date arithmatic
        period + 2 = '2017-03'
        pd.Timestamp('2017-01-31','M') + 1 = (2017-02-28, 'M')

    Sequence of dates & times
        index = pd.date_range(start,period,freq)
        index = [dates, freq='M']
        pd.DateTimeIndex() - sequence of timestamp objects

    you can convert values from timestamp to periods        
        index[0].to_period()

    Freq alias & time info
        period - hour, day, week, month,quarter,year - can be further diff by begg or end of the period or buss specific def
        pd.Timestamp - second,minute,hour,day,month,quarter,year,weekday,dayofweek,weekofyear,dayofyear

Sample Example
# Create the range of dates here
    seven_days = pd.date_range(start='2017-01-01',periods=7,freq='D')
# Iterate over the dates and print the number and name of the weekday
for day in seven_days:
    print(day.dayofweek, day.day_name()) 

Time series Transformation
    String to datetime64
    Selecting & slicing for subperiods
    Setting & changing Dattimeindex freq
        upsampling ( M- D ) & downsampling ( D - M) 
    
    Example data google stock data
        2 columns - dates & prices ( almost daily but could be missing )
        1) first convert date object to datetime - pd.to_datetime(google.date)
        2) .set_index() - setting index has advantages - this lets you treat the entire data as the timeseries data
        3) 1 e.g. u can plot - google.price.plot(title='')
        4) 2 e.g google['2015'].info - all entries for year 2015 
        5) 3 e.g. google['2015':'2017'].info - all entries for year 2015 
    
        .asfreq('D') - in the above u dont have freq info - use .asfreq to add freq info - upsampling with 'D'

Sample Example

data = pd.read_csv('nyc.csv')
# Convert the date column to datetime64
data['date'] = pd.to_datetime(data['date'])
# Set date column as index
data.set_index('date', inplace=True)
# Plot data
data.plot(subplots=True)
plt.show()

# Create dataframe prices here
prices = pd.DataFrame()
# Select data for each year and concatenate with prices here 
for year in ['2013', '2014','2015']:
    price_per_year = yahoo.loc[year, ['price']].reset_index(drop=True)
    price_per_year.rename(columns={'price': year}, inplace=True)
    prices = pd.concat([prices, price_per_year], axis=1)
# Plot prices
prices.plot()
plt.show()

# Inspect data
print(co.info())
# Set the frequency to calendar daily
co = co.asfreq('D')
# Plot the data
co.plot(subplots=True)
plt.show()
# Set frequency to monthly
co = co.asfreq('M')
# Plot the data
co.plot(subplots=True)
plt.show()

Manipulate Timeseries data
    Basic timeseries calculatiions
    Shift or lags values back / forward back in time
    get the difference in value for a given time period    
    Compute the % change over any number of periods

google = pd.read_csv(google.csv,par_dates=['date'],index_col='date')

.shift()?
    defaults to periods =1
    1 period into the future ( first value is null )
    gogle[shifted] = google.price.shift()
.shift(periods=-1)
    lagged data
    1 period back in time
    google[lagged] = google.price.shift(periods=-1)

    Calculations
        .div
            one period percent change - xt/xt-1
            google[change] = google.price.div(google.shifted)
            google[return] = google.change.sub(1).mul(100)

.diff()
    diff in value for two adjacent periods - xt - xt-1
    google[diff] = google.price.diff()

.pct_change()
    percentage change for two adjacent periods - xt / xt-1
    google[pct change] = google.price.pct_change().mul(100) - return & pct_change are the same

looking ahead e.g
    google[return_3d] = google.pct_change(periods=3).mul(100)
    percent change for 2 periods. 3 trading days apart


Sample example
# Import data here
google = pd.read_csv('google.csv', parse_dates=['Date'], index_col='Date')

# Set data frequency to business daily
google = google.asfreq('B')

print(google.head())

# Create 'lagged' and 'shifted'
google['lagged'] = google.Close.shift(periods=-90)
google['shifted'] = google.Close.shift(periods=90)

# Plot the google price series
google.plot(subplots=True)
plt.show()

# Created shifted_30 here
yahoo['shifted_30'] = yahoo.price.shift(periods=30)

# Subtract shifted_30 from price
yahoo['change_30'] = yahoo['price'] - yahoo['shifted_30']

# Get the 30-day price difference
yahoo['diff_30'] = yahoo.price.diff(periods=30)

# Inspect the last five rows of price
print(yahoo.tail())

# Show the value_counts of the difference between change_30 and diff_30
print(yahoo.change_30.sub(yahoo['diff_30']).value_counts())

# Create daily_return
google['daily_return'] = google.Close.pct_change(periods=1).mul(100)

# Create monthly_return
google['monthly_return'] = google.Close.pct_change(periods=30).mul(100)

# Create annual_return
google['annual_return'] = google.Close.pct_change(periods=360).mul(100)

# Plot the result
google.plot(subplots=True)
plt.show()

Growth Rates
    comparing stock performance
        normalize price series to start at 100
           Div all prices by furst in series, mul by 100
                same starting point
                all prices relative to starting point
                diff to starting point in % points

    Normalizing a single series
        sample google data - date ( index ) , Price
            first_price = google.iloc[0]
            normalized = google.price.div(first_price).mul(100)
    Normalizing multiple series
        say u have google,apple & meta
            first_price = stock.iloc[0]  - contains a series ( 3 first values )
            normalized = google.price.div(first_price).mul(100) - automatically div corresponding first price
    against a benchmark
        diff = normailized['tickers'].sub([normalized['sp500']],axis=0) - subtrack a series from each Dataframe column by alinging indexes

    Sample
        # Import data here
        prices = pd.read_csv('asset_classes.csv',parse_dates=['DATE'],index_col='DATE')

        # Inspect prices here
        print(prices.info())

        # Select first prices
        first_prices = prices.iloc[0]

        # Create normalized
        normalized = prices.div(first_prices).mul(100)

        # Plot normalized
        normalized.plot(subplots=True)
        plt.show()

        
        # Create tickers
        tickers = ['MSFT','AAPL']

        # Import stock data here
        stocks = pd.read_csv('msft_aapl.csv',parse_dates=['date'],index_col='date')

        # Import index here
        sp500 = pd.read_csv('sp500.csv',parse_dates=['date'],index_col='date')
        print(sp500.head())

        # Concatenate stocks and index here
        data = pd.concat([stocks,sp500],axis=1).dropna()
        print(data.head())

        # Normalize data
        normalized = data.div(data.iloc[0]).mul(100)
        print(normalized.head())

        # Subtract the normalized index from the normalized stock prices, and plot the result
        normalized[tickers].sub(normalized['SP500'],axis=0).plot()
        plt.show()

    Changing frequency
        .asfreq('')
        upsampling: fill or interpolate missing data
        Downsampleing: aggegate existing data
            .asfreq(),.reindex()
            .resample() + transformation method

        say u create quartly data and convert it to monthly freq; you create new rows whose values are missing
        for these missing value u can either ffill, bfill or fill with default values 

        monthly['ffill'] = quarterly.asfreq('M',method='ffill')

        upsampling & interpolation with .resample()
                .resample(): similar to groupby()
                groups data within resampling period and applies one or seveal methos to each group
                new date determined by offset start, end etc
                upsampling: ffill or bfill or inerpolate values
                downsampling: apply aggregation to existing data

                resample creates new date for freq offset
                calendar month end - M, MS - start of the month
                business month end - BM, BMS - start of the business month

                unrate.asfreq('MS') / unrate.resample('MS')

                interpolate gdp quartly data into monthly
                gdp1 = gdp.resample('MS').ffill().add_suffix('_ffill')

                gdp2 = gdp.resample('MS').interpolate().add_suffix('_inter')
                    .interpolate() : finds value on straight line betweeen existing data
                    e.g. 1.2 and 7.8 ( finds 2 values 3.4 & 5.6 between 1.0 & 7.2)

                # Inspect data here
                # 2010-01-01 to 2017-01-01 - monthly info
                print(monthly.info())
                
                # Create weekly dates
                weekly_dates = pd.date_range(start=monthly.index.min(),end=monthly.index.max(),freq='W')
                
                # Reindex monthly to weekly data
                weekly = monthly.reindex(weekly_dates)
                print(weekly.head(10))
                
                # Create ffill and interpolated columns
                weekly['ffill'] = weekly.UNRATE.ffill()
                weekly['interpolated'] = weekly.UNRATE.interpolate()
                
                # Plot weekly
                weekly.plot(subplots=True)
                plt.show()

        Downsampling & aggregation methods
            hour to day / day to month
            how to represent the existing values at the new date - aggregating
            sample:
                ozone = ozone.resample('D') 
                ozone.resample('M').mean() - monthly average assigned to the end of the month
                ozone.resample('M').median()
                ozone.resample('M').agg(['mean','std'])

                .first() - select first data point from each period

            matplotlib tip - matplotlib lets u plot again on the axes object returned by the first plot
                ozone = ozone.loc['2016':]
                ax=ozone.plot()
                monthly = ozone.resample('M').mean() 
                monthly.add_suffix('_monthly').plot(ax=ax)

    Window Functions
        Windows identify sub periods of ur time series
        calculate metrics for sub periods inside the window
        Create a new time series of metrics
        Two types of windows:
            Rolling: same size, sliding
            Expanding: grows and contain all prior values

        Rolling:
            data.rolling(window=30).mean()
                30 business days
                less than 30 - pandas doent calculate ; alter using min_periods and chose < 30
                30D - 30 calendar days

                r90 = data.rolling(window='90D').mean()
                data['mean90'] = r90
                data['r360'] = data['price'].rolling(window='360D').mean()

            Multiple rolling metrics
                rolling = data.google.rooling('360D')
                q10 = rolling.quantile(0.1).to_frame('q10')
                median = rolling.median.to_frame('q10')
                q90 = rolling.quantile(0.9).to_frame('q90')
                pd.concat([q10,median,q90],axis=1).plot()
        
        Expanding Window Functions:
            calculate metric for upto curr date
            new timeseries reflects all historical values
            usefull for running Rate of return, min/max
            two options
                .expanding()
                .cumsum(),.cumprod(),.cummin()

            df['exp'] = df.data.expanding().sum() / df.data.cumsum()

            rate of return - 
                rt = (Pt / Pt-1) -1  - .pct_change()
                Rt = (1+r1)(1+r2)()() - 1 - .cumprod()
                for basic math - .add(),.sb(),.mul,.div()

                pr = data.SP500.pct_change()
                pr_one = pr.add(1)
                cumm_ret = pr_one.cumprod().sub(1)

                Rolling annual rate of return
                    def multi_period_return(period_returns):
                        return np.prod(period_returns+1) - 1
                    pr = data.SP500.pct_change()
                    r = pr.rolling('360D').apply(multi_period_return)
                    data['rol_1yr_ret'] = r.mul(100)

    Random Walks
        Daily stock return are hard to predict
        Models often assume they are random in nature
        from random return to prices: use .cumprod()

        A random walk in time series is a statistical phenomenon where the value of a variable evolves over time in a manner that is unpredictable and depends solely on 
        its current value plus a random disturbance. It is often used as a simple model to represent stock prices, currency exchange rates, and other financial data.

        Key Features of a Random Walk:
        Lack of Predictability:
            The next value in the series is independent of past values, making the series non-deterministic.
            Changes are purely random, meaning no trends or cycles are present.
        Mathematical Representation: A random walk can be expressed as:
            Yt = Yt-1 + ϵt	
                ϵt : Random error term
 
        Examples of Random Walks:
            Stock Prices: Often modeled as random walks because price changes are hard to predict and influenced by new information.
            Brownian Motion: A physical analogy where particles move randomly in a fluid.
        Implications in Time Series Analysis:
            Forecasting: Since random walks are inherently unpredictable, traditional forecasting methods (e.g., moving averages) are ineffective without modifications.
            Differencing: To make the series stationary, taking the first difference is a common preprocessing step in models like ARIMA.            

        #Walk 1
        seed(42)
        random_walk = normal(loc=0.001, scale=0.01, size=2500)
        random_walk = pd.Series(random_walk)
        random_prices = random_walk.add(1).cumprod()
        random_prices.mul(1000).plot()
        plt.show()    
        #Walk 2
        seed(42)
        daily_returns = fb.pct_change().dropna()
        n_obs = daily_returns.count()
        random_walk = choice(daily_returns,size=n_obs)
        random_walk = pd.Series(random_walk)
        sns.distplot(random_walk)
        plt.show()        
        #Walk 3
        # Select fb start price here
        start = fb.price.first('D')
        random_walk = random_walk.add(1)
        random_price = start.append(random_walk)
        random_price = random_price.cumprod()
        fb['random'] = random_price
        fb['random'].plot()
        plt.show()

    Correlation & Time series
        Strength of linear relationship; Positive or negative
        Not non linear relationships

        daily_returns = data.pct_change()
        sns.jointplot(x='',y='',data=)

        correlations = return.corr()
        sns.heatmap(correlations, annot=True) 

        
    # Inspect data here
print(data.info())

# Calculate year-end prices here
annual_prices = data.resample('A').last()
print(annual_prices)

# Calculate annual returns here
annual_returns = annual_prices.pct_change()

# Calculate and print the correlation matrix here
correlations = annual_returns.corr()
print(correlations)

# Visualize the correlations as heatmap here
sns.heatmap(correlations,annot=True)
plt.show()
    
"""


