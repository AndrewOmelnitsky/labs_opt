import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from datetime import datetime, timedelta
import math
import functools


class PredictionModel:
    def __init__(self):
        self._template = np.zeros((7, 24))
        self._shop_id = -1
        self._data = None
        
        self.week_day_by_int = {
           0: "Понеділок",
           1: "Вівторок",
           2: "Середа",
           3: "Четвер",
           4: "П'ятниця",
           5: "Субота",
           6: "Неділя",
        }
        
    def count_by_day_and_hour(self):
        ...
                
    def _count_average_template(self):
        average = lambda sales: sales.mean()
        self._count_template_by_sales(average)
        
    def _count_median_template(self):
        average = lambda sales: sales.median()
        self._count_template_by_sales(average)
                
    def _count_template_by_sales(self, func):
        for i, day in enumerate(range(0, 7)):
            all_week_day_data = self._data[self._data['week_day'] == day]
            hours = all_week_day_data['time'].unique()
            
            for j, hour in enumerate(hours):
                all_data_in_hour = all_week_day_data[all_week_day_data['time'] == hour]
                sales = all_data_in_hour['All']
                
                self._template[i][j] = func(sales)  
        
    def _preparing_data_frame(self):
        self._data["All"] = self._data["All"].replace(np.nan, 0)
        weeks_day = []
        weeks_id = []
        week_id = 0
        last_week_day = 0
        for date in self._data['date']:
            date = datetime.strptime(date, "%Y-%m-%d").date()
            weeks_day.append(date.weekday())
            if date.weekday() == 0 and last_week_day == 6:
                week_id += 1
            weeks_id.append(week_id)
            last_week_day = date.weekday()
            
        self._data['week_day'] = weeks_day
        self._data['week_id'] = weeks_id
        
    def _training_by_type(self, training_type):
        if training_type == 'median':
            self._count_median_template()
        elif training_type == 'average':
            self._count_average_template()
        
    def training_model(self, data_frame, training_type='median'):
        self._data = data_frame
        self._preparing_data_frame()
        self._training_by_type(training_type)
        
    def _get_week_by_date(self, date):
        week = [date]
        
        date_now = date
        while True:
            # print(date_now, date_now.weekday())
            date_now = date_now + timedelta(days=1)
            if date_now.weekday() == 0:
                break
            
            week.append(date_now)
            
        date_now = date
        while True:
            # print(date_now, date_now.weekday())
            date_now = date_now - timedelta(days=1)
            if date_now.weekday() == 6:
                break
            
            week.insert(0, date_now)

        # print(date)
        # print(week)
        return week
        
    def plot_statistic(self, ploted_days='__all__'):
        if ploted_days == '__all__':
            ploted_days = range(0, 7)
            
        w = math.ceil(math.sqrt(len(ploted_days)))
        h = math.ceil(len(ploted_days) / w)
        
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        axes = []
        
        for i, day in enumerate(ploted_days):
            ax = plt.subplot(h, w, i+1)
            axes.append(ax)
            ax.set_title(self.week_day_by_int[day])
            ax.set_xlabel('Години')
            ax.set_ylabel('Продажі')

        for i, day in enumerate(ploted_days):
            all_week_day_data = self._data[self._data['week_day'] == day]
            weeks_dates = all_week_day_data['date'].unique()
            for week_date in weeks_dates:
                week_data = all_week_day_data[self._data['date'] == week_date]
                
                sales = week_data['All']
                hours = week_data['time']
                axes[i].plot(hours, sales, color='black')
                
        reserve_template = self._template.copy()
        
        self._training_by_type('median')
        for i, day in enumerate(ploted_days):
            sales = self._template[day]
            hours = range(0, 24)
            
            axes[i].plot(hours, sales, color='g', label='median')
            
        self._training_by_type('average')
        for i, day in enumerate(ploted_days):
            sales = self._template[day]
            hours = range(0, 24)
            
            axes[i].plot(hours, sales, color='r', label='average')
            
        self._template = reserve_template
            
        plt.legend()
        plt.show()
        
    def _count_k(self, week_day, data):
        if len(data) == 0:
            return 1
        if data.iloc[-1]['All'] == 0:
            return 1
        
        n = 0
        result = 0
        for hour_data_id in range(len(data)):
            hour_data = data.iloc[hour_data_id]
            if self._template[week_day][hour_data['time']] == 0:
                continue
            
            n += 1
            result += hour_data['All'] / self._template[week_day][hour_data['time']]
            
        if n == 0:
            return 1
        return result / n
        
    def predict(self, week_day, hour, history, sub_history=None, is_adapt_template=False, template_id=None):
        if hour == 0:
            if template_id is not None:
                template_id.append(week_day)
            return 0
        if sub_history is None:
            k = self._count_k(week_day, history)
        else:
            k = self._count_k(week_day, pd.concat([sub_history, history]))
            
        if is_adapt_template:
            
            sales = []
            for j in range(len(history)):
                t_h = history.iloc[:j]
                sales.append(self.predict(week_day, j, t_h, sub_history))
                    
            min_val = self.count_lost(list(history['All']), sales)
            t_id = week_day
            for i in range(self._template.shape[0]):
                sales = []
                for j in range(len(history)):
                    t_h = history.iloc[:j]
                    sales.append(self.predict(i, j, t_h, sub_history))
                    
                res = self.count_lost(list(history['All']), sales)
                # if hour < 12:
                #     print(f'{history["All"]=}')
                #     print(f'{sales=}')
                #     print(f'{res=}')
                #     print(f'{min_val=}')
                #     print(f'{t_id=}')
                
                # print('='*20)
                # print(f'{i=}')
                # print(f'{res=}')
                # print(f'{min_val=}')
                # print(f'{t_id=}')
                # print('='*20)
                if min_val > res:
                    min_val = res
                    t_id = i
                    
            return self.predict(t_id, hour, history, sub_history, template_id=template_id)
        else:
            if template_id is not None:
                template_id.append(week_day)
            template = self._template[week_day][hour]
            return k * template
    
    def count_lost(self, real, predicted):
        assert len(real) == len(predicted)
        
        lost = 0
        for i in range(len(real)):
            lost += abs(real[i] - predicted[i])
            
        return lost
    
    
    
def test_predict_for_template(df):
    df = df.copy()
    model = PredictionModel()
    # model.training_model(df, 'average')
    model.training_model(df)
    
    
    last_week_id = (df['week_id'].unique()[-2])
    losts = []
    for id_temp in range(last_week_id):
        t_data = df[df['week_id'] <= id_temp]
        r_data = df[df['week_id'] == last_week_id]
        # print(t_data)
        # print(r_data)
        
        
        model.training_model(t_data)
        
    
        day = 2
        # print(r_data)
        target_date = r_data[r_data['week_day'] == day].iloc[-1]['date']
        target_data = r_data[r_data['date'] == target_date]

        # print(f'{target_date=}')
        # print(target_data)
        
        y = []
        k = []
        sales = target_data
        hours = target_data['time']
        for t in range(len(sales)):
            y.append(model.predict(day, t, sales.iloc[:t]))
            k.append(model._count_k(day, sales.iloc[:t]))
            
        lost = model.count_lost(list(sales['All']), list(y))
        losts.append(lost)
        
    plt.xlabel('Кількість тижнів для навчання')
    plt.ylabel('Втрати')
    plt.plot(range(1, last_week_id+1), losts)
    plt.show()
    

def test_predict_for_k(df):
    df = df.copy()
    model = PredictionModel()
    # model.training_model(df, 'average')
    model.training_model(df)
    
    
    last_week_id = (df['week_id'].unique()[-1])
    t_data = df[df['week_id'] != last_week_id]
    r_data = df[df['week_id'] == last_week_id]
    # print(t_data)
    # print(r_data)
    
    
    model.training_model(t_data)
    

    day = 0
    # print(r_data)
    target_date = r_data[r_data['week_day'] == day].iloc[-1]['date']
    target_data = r_data[r_data['date'] == target_date]
    
    
    losts = []
    for id_temp in range(last_week_id+1):
        y = []
        k = []
        # history_added = df[(df['week_id'] <= id_temp) & (df['week_day'] == day)]
        history_added = df[(df['week_id'] < id_temp) & (df['week_day'] == day)]
        print(history_added)
        sales_traget = target_data['All']
        hours = target_data['time']
        for t, value in enumerate(sales_traget):
            sales = target_data.iloc[:t]
            # print(f'{history_added=}')
            # print(f'{sales=}')
            # print(f'{target_data.iloc[:t+1]=}')
            y.append(model.predict(day, t, sales, history_added))
            k.append(model._count_k(day, sales))
            
        lost = model.count_lost(list(sales_traget), list(y))
        losts.append(lost)
        
    plt.xlabel('Кількість тижнів для навчання')
    plt.ylabel('Втрати')
    plt.plot(range(last_week_id+1), losts)
    plt.show()
    


def get_target_func():
    df = pd.read_csv('data.csv', sep=';')
    df = df.set_index('index')
    # df_res = df.copy()    
    
    # df = df.copy()
    model = PredictionModel()
    model.training_model(df)
    
    day = 0
    last_week_id = (df['week_id'].unique()[-1])

        
    @functools.cache
    def func(id_temp, week_id):
        t_data = df[df['week_id'] < id_temp]
        r_data = df[df['week_id'] == last_week_id]
        
        model = PredictionModel()
        model.training_model(t_data)

        target_date = r_data[r_data['week_day'] == day].iloc[-1]['date']
        target_data = r_data[r_data['date'] == target_date]
        
        y = []
        history_added = df[(df['week_id'] < week_id) & (df['week_day'] == day)]
        sales_traget = target_data['All']
        # print(model._template)
        # print(t_data)
        # print(history_added)
        for t, value in enumerate(sales_traget):
            sales = target_data.iloc[:t]
            y.append(model.predict(day, t, sales, history_added))
            
        lost = model.count_lost(list(sales_traget), list(y))
        return lost
    
    def target_func(template_n, k_n):
        # print(template_n, k_n)
        template_n = round(template_n)
        k_n = round(k_n)
        return func(template_n, k_n)
        
    return target_func
        

func = get_target_func()
lost = []
print(func(7, 0))
# for i in range(0, 17):
#     lost.append(func(i, 0))
    
# print(min(lost), lost.index(min(lost)))
# print(lost[lost.index(min(lost))])
# 7