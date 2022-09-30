from finpy_tse import finpy_tse as fpy
import numpy as np
import torch
from nn.modules import translob

if __name__ == "__main__":

    stock1 = fpy.Get_IntradayOB_History(
        stock='کرمان',
        start_date='1401-06-01',
        end_date='1401-06-07',
        jalali_date=True,
        combined_datatime=False,
        show_progress=True)

    stock2 = fpy.Get_IntradayOB_History(
        stock='وخارزم',
        start_date='1401-06-01',
        end_date='1401-06-07',
        jalali_date=True,
        combined_datatime=False,
        show_progress=True)

    stock3 = fpy.Get_IntradayOB_History(
        stock='خودرو',
        start_date='1401-06-01',
        end_date='1401-06-07',
        jalali_date=True,
        combined_datatime=False,
        show_progress=True)

    stock4 = fpy.Get_IntradayOB_History(
        stock='فزرین',
        start_date='1401-06-01',
        end_date='1401-06-07',
        jalali_date=True,
        combined_datatime=False,
        show_progress=True)

    stock5 = fpy.Get_IntradayOB_History(
        stock='فگستر',
        start_date='1401-06-01',
        end_date='1401-06-07',
        jalali_date=True,
        combined_datatime=False,
        show_progress=True)

    stock1 = np.array(stock1)
    stock2 = np.array(stock2)
    stock3 = np.array(stock3)
    stock4 = np.array(stock4)
    stock5 = np.array(stock5)

    input_data = np.concatenate((stock1, stock2, stock3, stock4, stock5))
    input_tensor = torch.from_numpy(input_data)

    model = translob.TransLOB()
    output_tensor = model.forward(input_tensor)

    true_answers = 0.0
    train_size = 0.0
    for i in range(8, 11):
        start_date = '1401-06-0' + str(i)
        test1 = fpy.Get_Price_History(
            stock='کرمان',
            start_date=start_date,
            end_date=start_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False)
        test2 = fpy.Get_Price_History(
            stock='وخارزم',
            start_date=start_date,
            end_date=start_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False)
        test3 = fpy.Get_Price_History(
            stock='خودرو',
            start_date=start_date,
            end_date=start_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False)
        test4 = fpy.Get_Price_History(
            stock='فزرین',
            start_date=start_date,
            end_date=start_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False)
        test5 = fpy.Get_Price_History(
            stock='فگستر',
            start_date=start_date,
            end_date=start_date,
            ignore_date=False,
            adjust_price=False,
            show_weekday=False,
            double_date=False)

        test1 = np.array(test1)
        test2 = np.array(test2)
        test3 = np.array(test3)
        test4 = np.array(test4)
        test5 = np.array(test5)

        label_data = np.concatenate((test1, test2, test3, test4, test5))
        label_tensor = torch.from_numpy(label_data)

        comparison = torch.eq(output_tensor, label_tensor)
        true_answers += torch.sum(comparison)
        train_size += len(label_tensor)

    accuracy = (true_answers / train_size) * 100
    print("accuracy: ", accuracy)
