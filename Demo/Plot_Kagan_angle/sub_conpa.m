function [pa_con]=conpa(pa_flag,pa_beg,pa_end,pa_np)
pa_con=pa_flag*(pa_end-pa_beg)/pa_np+pa_beg;
end