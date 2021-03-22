import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np



#API 호출을 위한 라이브러리 임포트
import requests

from fbprophet import Prophet






#타임시리즈 데이터를 다뤄보자.
#야후금융에서 주식정보를 제공하는 라이브러리가 있다.
#yfinance라는 라이브러리 사용해서 주식정보를 불러오고 차트 그리는거 한다.

#해당 주식에 대한 트윗글들을 불러올 수 있는 API가 있다.
#stocktwits.com에서 제공하는 Restful API를 호출해서 데이터 가져오는 것 실습.
#이거 할줄 알면 API 다 가져다 쓸 수 있다. 네이버 파파고 이런 거.


def main():
    st.header('Online Stock Price Ticker')

    #yfinance 실행
    symbol = st.text_input('심볼 입력 : ')
    # symbol = 'AAPL'      #이거는 검색하면 나오는 거. 회사이름. 주식 보고 싶은 회사.
    data = yf.Ticker(symbol)

    today = datetime.now().date().isoformat()       #계속 최선정보가 들어가게
    print(today)


    df = data.history(start = '2010-06-01', end = '2021-03-22')    #문자열로 넣으면 알아서 가져온다/
    st.dataframe(df)

    st.subheader('종가')

    st.line_chart(df['Close'])  #종가

    st.subheader('거래량')
    st.line_chart(df['Volume'])  #거래량


    #yfinance 라이브러리만의 정보
    # data.info           #회사 정보, 얘는 딕셔너리
    # data.calendar         #얘는 데이터프레임
    # data.major_holders      #대주주
    # data.institutional_holders  #기간 정보
    # data.recommendations
    # data.dividends           #기간별 배당금
    div_df = data.dividends
    st.dataframe(div_df.resample('Y').sum()) #타임시리즈 데이터를 resample로 묶어서 사용할 수 있다. 다 했던 거임.
    
    new_df = div_df.reset_index()
    new_df['Year'] = new_df['Date'].dt.year     #이렇게 써야 프로펫할 때 사용할 수 있음. 인덱스가 시간이면 안됨. 컬럼으로 되어있어야함.
    st.dataframe(new_df)

    fig = plt.figure()
    plt.bar(new_df['Year'], new_df['Dividends'])  #x축 y축.   #연도별 배당금을 알 수 있다.
    st.pyplot(fig)

    #여러 주식 데이터를 한번에 보여주기.

    favorites = ['msft', 'tsla', 'nvda', 'aapl', 'amzn'] #멀티 셀렉트로 구현해도 되겠죠
    f_df = pd.DataFrame()

    for stock in favorites:    #페이보릿 안에 있는 걸 심볼로 넣으면 된다.
        f_df[stock] = yf.Ticker(stock).history(start = '2010-01-01', end = today)['Close']
    st.dataframe(f_df)
    st.line_chart(f_df)


    #리퀘스트를 하면 리스폰스를 준다.
 
    res = requests.get('https://api.stocktwits.com/api/2/streams/symbol/{}.json'.format(symbol))
 
    #JSON 형식이므로, .json() 이용.
    res_data = res.json()

    #파이썬의 딕셔너리와 리스트의 조합으로 사용가능
    # st.write(res_data)

    for message in res_data['messages']:

        #아바타는 왼쪽 글은 오른쪽에.
        col1, col2 = st.beta_columns([1,4])  #1:4의 비율로 컬럼 두개의 영역을 잡아달라. 컬럼 세개도 가능.

        with col1 :
            st.image(message['user']['avatar_url'])

        with col2 :
            st.write('유저 이름 : ' + message['user']['username'])
            st.write('트윗 내용 : ' + message['body'])

            st.write('올린 시간 : ' + message['created_at'])


    p_df = df.reset_index()    #인덱스에 있는 내용을 컬럼으로 옮기고
    p_df.rename(columns = {'Date': 'ds', 'Close':'y'}, inplace = True)
    # st.dataframe(p_df)

    #예측 가능

    m = Prophet()
    m.fit(p_df)
    future = m.make_future_dataframe(periods = 365)
    forecast = m.predict(future)
    st.dataframe(forecast)

    fig1 = m.plot(forecast)
    st.pyplot(fig1)
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    





    #리퀘스트 라이브러리 설치한다.
if __name__ == '__main__':
    main()