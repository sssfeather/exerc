%M02_06_fft_abs.m计算时间序列的DFT绝对值、
%按时间窗归一化的傅里叶谱振幅绝对值、
%傅里叶振幅谱密度绝对值
%傅里叶谱的真振幅
% 第一类信号：持续的周期信号
% x(t)=3*sin(2*pi*t)+1.5*sin(2*pi*5*t)+2
%dt=0.01s,N=1024,t=dt*(1:N)
clear all
dt=0.02
%N=1024
%N=4096
%N=128
N=1000
t=dt*(1:N);
df=1/(N*dt)
f=df*(1:N)-df;
x=4*sin(2*pi*t)+2*sin(2*pi*5*t)+2;
y=fft(x);
dft_abs=abs(y);
dft_abs(1)=dft_abs(1)/2
fn_abs=abs(y/N);
fn_abs(1)=fn_abs(1)/2
fnd_abs=abs(dt*y);
fnd_abs(1)=fnd_abs(1)/2
an=2*abs(y)/N;
an(1)=an(1)/2
N1=4000
t1=dt*(1:N1);
df1=1/(N1*dt)
f1=df1*(1:N1)-df1;
x1=4*sin(2*pi*t1)+2*sin(2*pi*5*t1)+2;
%x=4*sin(2*pi*t);
figure(1)
%subplot(5,1,1)
plot(t1(1:N1),x1(1:N1))
xlabel('t/s')
title('(a)     test signal : x=4*sin(2*pi*t)+2*sin(2*pi*5*t)+2')
figure(2)
y1=fft(x1);
dft_abs1=abs(y1);
dft_abs1(1)=dft_abs1(1)/2
subplot(221)
bar(f(1:N),dft_abs(1:N),0.2)
%axis([-1 df*N 0 5000])
axis([-1 df*N 0 10000])
title('(b)        dft-abs')
xlabel('f/Hz')
%------------------------------------
subplot(222)
bar(f1(1:N1),dft_abs1(1:N1),0.2)
axis([-1 df1*N1 0 10000])
title('(b)        dft-abs')
xlabel('f/Hz')
%-------------------------------------
fn_abs1=abs(y1/N1);
fn_abs1(1)=fn_abs1(1)/2
subplot(223)
bar(f(1:N),fn_abs(1:N),0.2)
axis([-1 df*N 0 4])
title('(c)        fn-abs')
xlabel('f/Hz')
subplot(224)
bar(f1(1:N1),fn_abs1(1:N1),0.2)
axis([-1 df1*N1 0 4])
title('(c)        fn-abs')
xlabel('f/Hz')
%------------------------------------
figure(3)
fnd_abs1=abs(dt*y1);
fnd_abs1(1)=fnd_abs1(1)/2
subplot(221)
bar(f(1:N),fnd_abs(1:N),0.2)
axis([-1 df*N 0 200])
title('(d)         fnd-abs')
xlabel('f/Hz')
subplot(222)
bar(f1(1:N1),fnd_abs1(1:N1),0.2)
axis([-1 df1*N1 0 200])
title('(d)         fnd-abs')
xlabel('f/Hz')
an1=2*abs(y1)/N1;
an1(1)=an1(1)/2
subplot(223)
bar(f(1:N),an(1:N),0.2)
axis([-1 df*N 0 5])
title('(e)            An')
xlabel('f/Hz')
subplot(224)
bar(f1(1:N1),an1(1:N1),0.2)
axis([-1 df1*N1 0 5])
title('(e)            An')
xlabel('f/Hz')
%-----------------------------------
%使用地震记录波形
%N=1024
N1=16000
N=8000
%N=8192
load M02_04_dalianE1.txt; % 读入一道波形数据
x1(1:N1)=M02_04_dalianE1(2501:N1+2500);
x(1:N)=M02_04_dalianE1(2501:N+2500);
dt=0.02
t=dt*(1:N);
df=1/(N*dt)
f=df*(1:N)-df;
y=fft(x);
dft_abs=abs(y);
dft_abs(1)=dft_abs(1)/2
fn_abs=abs(y/N);
fn_abs(1)=fn_abs(1)/2
fnd_abs=abs(dt*y);
fnd_abs(1)=fnd_abs(1)/2
an=2*abs(y)/N;
an(1)=an(1)/2
t1=dt*(1:N1);
df1=1/(N1*dt)
f1=df1*(1:N1)-df1;
%x=4*sin(2*pi*t);
figure(4)
plot(t1(1:N1),x1(1:N1))
xlabel('t/s')
title('(a)     test signal : x=M02-04-dalianE1(2501:N+2500)')
figure(5)
y1=fft(x1);
dft_abs1=abs(y1);
dft_abs1(1)=dft_abs1(1)/2
subplot(221)
bar(f(1:N),dft_abs(1:N),0.2)
%axis([-1 df*N 0 5000])
axis([-1 df*N 0 8000000])
title('(b)        dft-abs')
xlabel('f/Hz')
%------------------------------------
subplot(222)
bar(f1(1:N1),dft_abs1(1:N1),0.2)
axis([-1 df1*N1 0 8000000])
title('(b)        dft-abs')
xlabel('f/Hz')
%-------------------------------------
fn_abs1=abs(y1/N1);
fn_abs1(1)=fn_abs1(1)/2
subplot(223)
bar(f(1:N),fn_abs(1:N),0.2)
axis([-1 df*N 0 1200])
title('(c)        fn-abs')
xlabel('f/Hz')
subplot(224)
bar(f1(1:N1),fn_abs1(1:N1),0.2)
axis([-1 df1*N1 0 1200])
title('(c)        fn-abs')
xlabel('f/Hz')
%------------------------------------
figure(6)
fnd_abs1=abs(dt*y1);
fnd_abs1(1)=fnd_abs1(1)/2
subplot(221)
bar(f(1:N),fnd_abs(1:N),0.2)
axis([-1 df*N 0 200000])
title('(d)         fnd-abs')
xlabel('f/Hz')
subplot(222)
bar(f1(1:N1),fnd_abs1(1:N1),0.2)
axis([-1 df1*N1 0 200000])
title('(d)         fnd-abs')
xlabel('f/Hz')
an1=2*abs(y1)/N1;
an1(1)=an1(1)/2
subplot(223)
bar(f(1:N),an(1:N),0.2)
axis([-1 df*N 0 1500])
title('(e)            An')
xlabel('f/Hz')
subplot(224)
bar(f1(1:N1),an1(1:N1),0.2)
axis([-1 df1*N1 0 1500])
title('(e)            An')
xlabel('f/Hz')
%----------------------------------------------------------
