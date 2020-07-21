import scipy.io.wavfile as wav
import scipy.fftpack as fft
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import wave
import pandas as pd

name = input("분석을 할 파일을 exe와 같은 경로에 두세요. 파일의 이름은?(확장명 포함) : ")

#필요한 모듈들을 불러들인다.

inwav = wav.read(name)
pyinwav = wave.open(name)
frames = pyinwav.getnframes()
rate = pyinwav.getframerate()

print(frames, rate)

wavarr = np.array(inwav[1], dtype=float)
if name == "sine.wav":
    wavarr = map(lambda x : x[0],wavarr.tolist())
    wavarr = np.array(list(wavarr))

print(wavarr)
#wav파일을 읽고, 이를 실수 리스트 형태로 불러들인다. 확인 결과, frame/rate초의 녹음파일 속에 frame개의 샘플이 들어있었다.

x_space = fft.fftfreq(len(wavarr))
#샘플의 수를 넣으면 x축을 자동으로 fft가 반환하는 진동수로 바꿔준다.
x_space = x_space / (1/rate)
#여기서 반환하는 진동수는 사실상 1s가 아니라 1sample인데, 1sample 당 1/rate 만큼의 실제 시간간격이기에 이를 보정한다.




wavarr = fft.fft(wavarr) / len(wavarr)
#fft를 적용하고 적당히 줄인다.
wavarr_mag = abs(wavarr)
#fft결과는 복소수이므로 우리는 크기만을 취한다.(위상자 X)

mask = int( len(x_space) / 2 )
x_space_cut = x_space[:mask]
wavarr_mag_cut = wavarr_mag[:mask] * 2
#파이썬의 fft 결과는 음의 진폭은 음의 진동수로 취급하여 나타내므로, 음의 진동수까지 양의 진동수에 합산해주기 위하여
#음의 진동수 부분을 자르고 양의 진동수 부분을 2배함.

#그냥 이렇게만 하면 그래프가 상당히 지저분해진다. 조금 smoothing 을 해주어야 한다.
wavarr_mag_smoothed = savgol_filter(wavarr_mag_cut, 51, 3)
for i in range(100):
    wavarr_mag_smoothed = savgol_filter(wavarr_mag_smoothed, 51, 3)


#그래프 미분.
def slope(x1,x2,y1,y2):
    return (y2 - y1) / (x2 - x1)

wavarr_mag_diff = np.array([])
for i in range(len(x_space_cut)):
    print(i)
    if i == len(x_space_cut) - 1:
        wavarr_mag_diff = np.append(wavarr_mag_diff,[1])
        break
    wavarr_mag_diff = np.append(wavarr_mag_diff, [slope(x_space_cut[i],x_space_cut[i+1],wavarr_mag_smoothed[i],wavarr_mag_smoothed[i+1])])

#미분 후 극점 찾기
x_space_maxi = np.array([])
wavarr_mag_maxi = np.array([])
for i in range(len(x_space_cut) - 1):
    yi = wavarr_mag_diff[i]
    yii = wavarr_mag_diff[i+1]
    if yi > 0 and yii < 0:
        if wavarr_mag_smoothed[i+1] > 5:
            x_space_maxi = np.append(x_space_maxi, [x_space_cut[i + 1]])
            wavarr_mag_maxi = np.append(wavarr_mag_maxi, [wavarr_mag_smoothed[i + 1]])

#적절히 값 조정하기
wavarr_mag_maxi_sort = np.sort(wavarr_mag_maxi)[::-1]
wavarr_mag_maxi_sort = wavarr_mag_maxi_sort[0:20]
x_space_maxi_sort = np.array([])
for value in wavarr_mag_maxi_sort:
    ii = np.where(wavarr_mag_maxi == value)[0][0]
    xi = x_space_maxi[ii]
    x_space_maxi_sort = np.append(x_space_maxi_sort,[xi])

print("done!")

output_pandas = pd.DataFrame(np.array([x_space_maxi_sort,wavarr_mag_maxi_sort]))
print(output_pandas)
output_pandas.to_csv(name.split('.')[0] + '.csv')

#plt.scatter(x_space_sort,wavrr_mag_sort)
plt.subplot(2,1,1)
plt.plot(x_space_cut,wavarr_mag_smoothed)
plt.scatter(x_space_maxi_sort, wavarr_mag_maxi_sort)
plt.xlabel("Hz")
plt.ylabel("fft amount")
plt.subplot(2,1,2)
plt.scatter(x_space_maxi_sort, wavarr_mag_maxi_sort)
plt.stem(x_space_maxi_sort, wavarr_mag_maxi_sort)
plt.xlabel("Hz")
plt.ylabel("fft amount")
# 그래프를 그린다.
plt.show()

