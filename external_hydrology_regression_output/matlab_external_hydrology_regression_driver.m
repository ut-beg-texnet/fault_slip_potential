try
addpath('C:/texnetwebtools/tools/fsp_python/external_hydrology_regression_output');
addpath('C:/texnetwebtools/tools/fsp_python/reference_old_code/support code');
fid_model = fopen('C:/texnetwebtools/tools/fsp_python/external_hydrology_regression_output/matlab_model_input.csv');
readInData = textscan(fid_model,'%s%s%s%s','Delimiter',',','Headerlines',1,'multipledelimsasone',true);
fclose(fid_model);
hDV = RegressionHDV();
hDV.hfig = [];
hDV.data = struct();
hDV.data.reservoir = struct();
hDV.data.reservoir.stringsImportedHydrology = [readInData{:,:}];
SpreadsheetStrings2HydrologyData(hDV.data.reservoir.stringsImportedHydrology, hDV);
numbersImportedHydrology = hDV.data.reservoir.numbersImportedHydrology;
yearsRepresentedHydroImport = hDV.data.reservoir.yearsRepresentedHydroImport;
faults = csvread('C:/texnetwebtools/tools/fsp_python/external_hydrology_regression_output/matlab_faults_xy.csv',1,0);
fid_out = fopen('C:/texnetwebtools/tools/fsp_python/external_hydrology_regression_output/matlab_fault_pressure.csv','w');
fprintf(fid_out,'year,fault_index,pressure_psi\n');
for year_index = 1:length(yearsRepresentedHydroImport)
    ts = yearsRepresentedHydroImport(year_index);
    thisYearData = numbersImportedHydrology(numbersImportedHydrology(:,4)==ts,:);
    thisYearX = thisYearData(:,1);
    thisYearY = thisYearData(:,2);
    thisYearPSI = thisYearData(:,3);
    fault_rows = faults(faults(:,1)==ts,:);
    fault_x = fault_rows(:,2);
    fault_y = fault_rows(:,3);
    ppOnFault = griddata(thisYearX,thisYearY,thisYearPSI,fault_x,fault_y);
    for k = 1:numel(ppOnFault)
        fprintf(fid_out,'%.16g,%d,%.16g\n',ts,k,ppOnFault(k));
    end
end
fclose(fid_out);
catch ME
fid = fopen('C:/texnetwebtools/tools/fsp_python/external_hydrology_regression_output/matlab_error.txt','w');
fprintf(fid,'%s\n',ME.message);
fclose(fid);
end
exit
