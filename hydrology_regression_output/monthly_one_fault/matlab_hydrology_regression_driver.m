try
addpath('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault');
addpath('C:/Users/bakirtzisn/Desktop/FSP_dev_test/matlab_code/support code');
addpath('C:/Users/bakirtzisn/Desktop/FSP_dev_test/matlab_code/technical code');
fault_x = [2.066604728573997];
fault_y = [2.416824742266897];

fid_wells = fopen('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault/matlab_wells_input.csv');
readInData = textscan(fid_wells,'%s%s%s%s%s%s','Delimiter',',','Headerlines',1);
fclose(fid_wells);
hDV = RegressionHDV();
hDV.hfig = [];
hDV.data = struct();
hDV.data.nwells_max = 10000;
hDV.data.realWellData = struct();
hDV.data.realWellData.columnIsNumber = logical([0,1,1,1,1,1]);
hDV.data.realWellData.stringsWellDataAdvanced = [readInData{:,:}];
hDV.data.realWellData.extrapolateInjectionCheck = 0;
SpreadsheetStrings2WellData(hDV.data.realWellData.stringsWellDataAdvanced, hDV);
if 0
    for k = 1:hDV.data.nwells
        hDV.data.realWellData.datenumBarrelsPerDay{k}(end,1) = datenum(2026,1,1,0,1,0);
    end
end
nwells = hDV.data.nwells;
well_x = hDV.data.realWellData.XEasting;
well_y = hDV.data.realWellData.YNorthing;
well_names = hDV.data.realWellData.wellNames;
d = hDV.data.realWellData.datenumBarrelsPerDay;

selected_index = 1;
for k = 1:nwells
    if strcmp(char(well_names{k}), 'SYN_SWD_01')
        selected_index = k;
        break;
    end
end
h = 100 * 0.3048;
porosity_percent = 10;
phi = porosity_percent / 100;
kap = 200;
rho = 1000;
mu = 0.0008;
beta = 3.6e-10;
alpha = 1.08e-09;
[S,T] = calcST([0,0,0,0,rho,9.81,mu,beta,alpha],h,phi,kap);
fault_pressure = zeros(length(fault_x),1);
for k = 1:nwells
    r_fault = sqrt((fault_x - well_x(k)).^2 + (fault_y - well_y(k)).^2) .* 1000;
    fault_pressure = fault_pressure + pfront(r_fault,2026,d{k},S,T,rho,9.81) .* 14.5;
end
csvwrite('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault/matlab_fault_pressure.csv',fault_pressure);
radial_km = [0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.7999999999999999;0.8999999999999999;0.9999999999999999;1.1;1.2;1.3;1.4;1.5;1.6;1.7;1.8;1.9;2;2.1;2.2;2.3;2.4;2.5;2.6;2.7;2.8;2.9;3;3.1;3.2;3.3;3.4;3.5;3.6;3.7;3.8;3.9;4;4.1;4.199999999999999;4.299999999999999;4.399999999999999;4.499999999999999;4.6;4.699999999999999;4.799999999999999;4.899999999999999;4.999999999999999;5.1;5.199999999999999;5.299999999999999;5.399999999999999;5.499999999999999;5.599999999999999;5.699999999999999;5.799999999999999;5.899999999999999;5.999999999999999;6.099999999999999;6.199999999999999;6.299999999999999;6.399999999999999;6.499999999999999;6.599999999999999;6.699999999999999;6.799999999999999;6.899999999999999;6.999999999999999;7.099999999999999;7.199999999999999;7.299999999999999;7.399999999999999;7.499999999999999;7.599999999999999;7.699999999999999;7.799999999999999;7.899999999999999;7.999999999999999;8.1;8.199999999999999;8.299999999999999;8.399999999999999;8.499999999999998;8.6;8.699999999999999;8.799999999999999;8.899999999999999;8.999999999999998;9.1;9.199999999999999;9.299999999999999;9.399999999999999;9.499999999999998;9.6;9.699999999999999;9.799999999999999;9.899999999999999;9.999999999999998;10.1;10.2;10.3;10.4;10.5;10.6;10.7;10.8;10.9;11;11.1;11.2;11.3;11.4;11.5;11.6;11.7;11.8;11.9;12;12.1;12.2;12.3;12.4;12.5;12.6;12.7;12.8;12.9;13;13.1;13.2;13.3;13.4;13.5;13.6;13.7;13.8;13.9;14;14.1;14.2;14.3;14.4;14.5;14.6;14.7;14.8;14.9;15;15.1;15.2;15.3;15.4;15.5;15.6;15.7;15.8;15.9;16;16.1;16.2;16.3;16.4;16.5;16.6;16.7;16.8;16.9;17;17.1;17.2;17.3;17.4;17.5;17.6;17.7;17.8;17.9;18;18.1;18.2;18.3;18.4;18.5;18.6;18.7;18.8;18.9;19;19.1;19.2;19.3;19.4;19.5;19.6;19.7;19.8;19.9;20];
radial_pressure = pfront(radial_km .* 1000,2026,d{selected_index},S,T,rho,9.81) .* 14.5;
csvwrite('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault/matlab_radial_pressure.csv',[radial_km,radial_pressure]);
fid_rates = fopen('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault/matlab_well_rates.csv','w');
fprintf(fid_rates,'well_id,datenum,rate_bbl_day\n');
for k = 1:nwells
    series = d{k};
    name = char(well_names{k});
    for r = 1:size(series,1)
        fprintf(fid_rates,'%s,%.16g,%.16g\n',name,series(r,1),series(r,2));
    end
end
fclose(fid_rates);


catch ME
fid = fopen('C:/texnetwebtools/tools/fsp_python/hydrology_regression_output/monthly_one_fault/matlab_error.txt','w');
fprintf(fid,'%s\n',ME.message);
fclose(fid);
end
exit
