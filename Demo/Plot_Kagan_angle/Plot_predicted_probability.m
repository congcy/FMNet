%
clear;
clc;
close all;

count=0;
numev=972;
for i=1:200:numev
        filename_test=sprintf('../test_output/predict_%06d.mat',i);
        load(filename_test);
        
        count_fig=0;
        figure;
        for k=1:3
            amp_true=true(:,:,k);
            amp_pred=pred(:,:,k);
            count_fig=count_fig+1;
            subplot(3,1,count_fig);
            plot(amp_true,'linewidth',2,'color','k');
            hold on;
            plot(amp_pred,'linewidth',2,'color','r');
            xlim([0 length(amp_true)]);
            ylim([0 1]);
            set(gca,'xticklabel',[]);
            switch k
                case 1
                   xlabel('strike');
                case 2
                   xlabel('dip');
                case 3
                   xlabel('rake');
            end
            ylabel('Probability');
            legend('true','predicted');
        end
        set(gcf,'unit','centimeters','position',[5 2 18 12]);
end


