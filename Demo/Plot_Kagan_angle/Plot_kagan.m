%
clear;
clc;
close all;

np_out=[128,128,128];

strike_beg=0;
strike_end=360;
dip_beg=0;
dip_end=90;
rake_beg=-90;
rake_end=90;

numev=972;
for i=1:numev
        filename_test=sprintf('../test_output/predict_%06d.mat',i);
        load(filename_test);
        
        % get the index of maximum values
        true_flag_strike=find(true(:,:,1)==max(true(:,:,1)));
        true_flag_dip=find(true(:,:,2)==max(true(:,:,2)));
        true_flag_rake=find(true(:,:,3)==max(true(:,:,3)));
        pred_flag_strike=find(pred(:,:,1)==max(pred(:,:,1)));
        pred_flag_dip=find(pred(:,:,2)==max(pred(:,:,2)));
        pred_flag_rake=find(pred(:,:,3)==max(pred(:,:,3)));

        % convert index to strike, dip, and rake
        true_strike=sub_conpa(true_flag_strike,strike_beg,strike_end,np_out(1));
        true_dip=sub_conpa(true_flag_dip,dip_beg,dip_end,np_out(2));
        true_rake=sub_conpa(true_flag_rake,rake_beg,rake_end,np_out(3));
        pred_strike=sub_conpa(pred_flag_strike,strike_beg,strike_end,np_out(1));
        pred_dip=sub_conpa(pred_flag_dip,dip_beg,dip_end,np_out(2));
        pred_rake=sub_conpa(pred_flag_rake,rake_beg,rake_end,np_out(3));
        
        % calculate the Kagan angle (subroutine is from by matlab open source library)
        [rotangle,theta,phi]=sub_kang([true_strike,true_dip,true_rake],[pred_strike,pred_dip,pred_rake]);
        kang(i)=rotangle;
end

% calculate percentage (kagan angle <= 20 degree)
kang=real(kang);
num1_kang=find(kang<=20);
num2_kang=find(kang>20);
fprintf('Good is %f,  bad is %f\n',length(num1_kang)/numev,length(num2_kang)/numev);

%% plot the histogram of Kagan angles
figure;
histogram(kang,100);
axis([-0.5 121 0 450]);
xlabel('Kagan angles','fontweight','bold');
ylabel('Counts','fontweight','bold');
set(findobj('FontSize',10),'FontSize',14,'fontweight','bold');
set(gcf,'unit','centimeters','position',[5 2 22 14]);

