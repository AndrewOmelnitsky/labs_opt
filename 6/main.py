import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math


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
                all_data_in_hour = all_week_day_data[self._data['time'] == hour]
                sales = all_data_in_hour['All']
                
                self._template[i][j] = func(sales)  
        
    def _preparing_data_frame(self):
        self._data["All"] = self._data["All"].replace(np.nan, 0)
        weeks_day = []
        for date in self._data['date']:
            date = datetime.strptime(date, "%Y-%m-%d").date()
            weeks_day.append(date.weekday())
            
        self._data['week_day'] = weeks_day
        
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
            if date_now.weekday() == 0:
                break
            
            date_now = date_now + timedelta(days=1)
            week.append(date_now)
            
        date_now = date
        while True:
            if date_now.weekday() == 0:
                break
            
            date_now = date_now - timedelta(days=1)
            week.insert(0, date_now)

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
            
            axes[i].plot(hours, sales, color='g')
            
        self._training_by_type('average')
        for i, day in enumerate(ploted_days):
            sales = self._template[day]
            hours = range(0, 24)
            
            axes[i].plot(hours, sales, color='r')
            
        self._template = reserve_template
        
        # target_date = self._data[self._data['week_day'] == 6]['date'].unique()[-1]
        # week = self._get_week_by_date(datetime.strptime(target_date, "%Y-%m-%d").date())
        # week = [d.strftime("%Y-%m-%d") for d in week]
        # print(week)
        # for i, day in enumerate(ploted_days):
        #     all_week_day_data = self._data[self._data['week_day'] == day]
        #     target_data = self._data[self._data['date'] == week[day]]
        #     print(target_date)
        #     print(target_data)
            
        #     y = []
        #     k = []
        #     sales = target_data['All']
        #     hours = target_data['time']
        #     for t, value in enumerate(sales):
        #         y.append(self.predict(day, t, sales[:t+1]))
        #         k.append(self._count_k(day, sales[:t+1]))
                
        #     ax = axes[i].twinx()
        #     ax.set_ylim(-1.5, 1.5)
        #     mark = '.'
        #     axes[i].plot(hours, sales, marker=mark, label='Реальні', color='r')
        #     axes[i].plot(hours, y, marker=mark, label='Прогноз', color='black')
        #     axes[i].plot(hours, self._template[day], marker=mark, label='Шаблон', color='b')
        #     ax.plot(hours, k, marker=mark, label='Коефіцієнт k', color='g')
        #     axes[i].legend()
        #     ax.legend()
            
        plt.show()
        
    def _count_k(self, week_day, data):
        if data.iloc[-1] == 0:
            return 1
        
        n = 0
        result = 0
        for i, p in enumerate(data):
            if self._template[week_day][i] == 0:
                continue
            
            n += 1
            result += p / self._template[week_day][i]
            
        return result / n
        
    def predict(self, week_day, hour, history):
        k = self._count_k(week_day, history)
        return k * self._template[(week_day + 1) % 7][hour]
        


def main():
    df = pd.read_csv('data1.csv', sep=';')
    df = df.set_index('index')
    
    model = PredictionModel()
    # model.training_model(df, 'average')
    model.training_model(df)
    # model.plot_statistic([0, 1, 2, 6,])
    model.plot_statistic('__all__')


if __name__ == '__main__':
    main()