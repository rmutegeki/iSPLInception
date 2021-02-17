function varargout = label_panorama(varargin)
% LABEL_PANORAMA MATLAB code for label_panorama.fig
%      LABEL_PANORAMA, by itself, creates a new LABEL_PANORAMA or raises the existing
%      singleton*.
%
%      H = LABEL_PANORAMA returns the handle to a new LABEL_PANORAMA or the handle to
%      the existing singleton*.
%
%      LABEL_PANORAMA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LABEL_PANORAMA.M with the given input arguments.
%
%      LABEL_PANORAMA('Property','Value',...) creates a new LABEL_PANORAMA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before label_panorama_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to label_panorama_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help label_panorama

% Last Modified by GUIDE v2.5 16-May-2012 11:05:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @label_panorama_OpeningFcn, ...
                   'gui_OutputFcn',  @label_panorama_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function dataset = load_dataset(filepath, filenames)
dataset = struct([]);
dataset(end+1).path = filepath;
for k = 1:3
    temp = importdata([filepath filenames{k}]);
    
    if isnumeric(temp)
        dataset.data = temp;
        dataset.data_file = filenames{k};
    else
        temp = {};
        f = fopen([filepath filenames{k}], 'r');
        oneline = fgetl(f);
        while(ischar(oneline))
            temp{end+1} = oneline;
            oneline = fgetl(f);
        end
        fclose(f);
        if any(ismember(temp, 'Label columns: '))
            dataset.column_legend = temp;
            dataset.column_legend_file = filenames{k};
            label_column_pointer = find(ismember(temp, 'Label columns: '));
        else
            dataset.label_legend = temp;
            dataset.label_legend_file = filenames{k};
        end
    end
end

dataset.label_columns = [];
dataset.label_columns_desc = {};
dataset.data_columns = [];
dataset.data_desc = {};
dataset.time_columns = [];
dataset.time_desc = {};

for k = label_column_pointer + 1 : length(dataset.column_legend)
    if ~isempty(strfind(upper(dataset.column_legend{k}), 'COLUMN:'))
        tmp = textscan(dataset.column_legend{k}, '%s');
        dataset.label_columns(end+1) = str2double(tmp{1}{2});
        dataset.label_columns_desc{end+1} = tmp{1}{3};
    end
end

for k = 1:label_column_pointer - 1
    if ~isempty(strfind(upper(dataset.column_legend{k}), 'COLUMN:'))
        if isempty(strfind(dataset.column_legend{k}, 'SEC')) % not a timestamp column
            dataset.data_columns(end+1) = sscanf(dataset.column_legend{k}, 'Column: %d');
            index = strfind(dataset.column_legend{k}, num2str(dataset.data_columns(end)));
            index = index + length(num2str(dataset.data_columns(end))) + 1;
            dataset.data_desc{end+1} = dataset.column_legend{k}(index:end);
        else % timestamp column
            dataset.time_columns(end+1) = sscanf(dataset.column_legend{k}, 'Column: %d');
            index = strfind(dataset.column_legend{k}, num2str(dataset.time_columns(end)));
            index = index + length(num2str(dataset.time_columns(end))) + 1;
            dataset.time_desc{end+1} = dataset.column_legend{k}(index:end);
        end
    end
end

dataset.label_id2name = cell(1,length(dataset.label_columns));
dataset.labels = cell(1,length(dataset.label_columns));
dataset.labels_and_nullclass = cell(1,length(dataset.label_columns));
dataset.label_id2img_coord = cell(1,length(dataset.label_columns));
dataset.label_img_coord2id = cell(1,length(dataset.label_columns));
dataset.label_image = cell(1,length(dataset.label_columns));

dataset.timestamps = zeros(size(dataset.data,1), 1); % prepare column for converted timestamps (in seconds)
for k = 1:length(dataset.time_columns)
    if strcmpi(dataset.time_desc{k}, 'SEC')
        dataset.timestamps = dataset.timestamps + dataset.data(:, dataset.time_columns(k));
    elseif strcmpi(dataset.time_desc{k}, 'MILLISEC')
        dataset.timestamps = dataset.timestamps + dataset.data(:, dataset.time_columns(k)) * .001;
    elseif strcmpi(dataset.time_desc{k}, 'MICROSEC')
        dataset.timestamps = dataset.timestamps + dataset.data(:, dataset.time_columns(k)) * .000001;
    else
        error(['Unrecognized time format: ' upper(dataset.time_desc{k}) '. Allowed formats: SEC, MILLISEC, MICROSEC.']);
    end
end


% id mappings
for c = 1:length(dataset.label_columns)
    dataset.label_id2name{c} = containers.Map('KeyType', 'double', 'ValueType', 'char');
    
    for k = 1:length(dataset.label_legend)
        id = sscanf(dataset.label_legend{k}, '%d -');
        if ~isempty(id)
            index = strfind(dataset.label_legend{k}, '-');
            if length(index) ~= 2
                error('Invalid format: "expected ID - LabelTrack - LabelName"');
            end
            tmp = textscan(dataset.label_legend{k}(index(1)+1:index(2)-1), '%s');
            if strcmp(tmp{1}{1}, dataset.label_columns_desc{c})
                dataset.label_id2name{c}(id) = underscore_clean(sscanf(dataset.label_legend{k}(index(2)+1 : end), '%s'));
            end
        end
    end
    
    dataset.label_id2img_coord{c} = containers.Map(keys(dataset.label_id2name{c}), 1:length(keys(dataset.label_id2name{c})));
    dataset.label_img_coord2id{c} = containers.Map(1:length(keys(dataset.label_id2name{c})), keys(dataset.label_id2name{c}));
    
    dataset.label_image{c} = zeros(1,length(keys(dataset.label_id2img_coord{c})));
    
    dataset.data(1, dataset.label_columns(c)) = 0; % just to be able to use everywhere (also at the very beginning and at the very end)...
    dataset.data(end, dataset.label_columns(c)) = 0; % ...the derivative of the label track to get correct start and stop times
    
    % dataset.labels is a matrix with following columns: 1 = ID, 2 = row in data
    % matrix where label starts, 3 = row in data matrix where label ends
    derivative = [0; diff(dataset.data(:, dataset.label_columns(c)))];
    starting_logicals = derivative ~= 0 & dataset.data(:, dataset.label_columns(c)) ~= 0;
    ending_logicals = derivative ~= 0;
    ending_logicals = [ending_logicals(2:end); 0] & dataset.data(:, dataset.label_columns(c)) ~= 0;
    dataset.labels{c} = [dataset.data(starting_logicals, dataset.label_columns(c)) find(starting_logicals) find(ending_logicals)];
    
    dataset.labels_and_nullclass{c} = zeros(1+2*size(dataset.labels{c},1),3);
    dataset.labels_and_nullclass{c}(2:2:end,:) = dataset.labels{c}; % insert null class between existing labels (interleaved)
    
    % null class starting at successive row after each label end; first null
    % starts at row 1; null class ending at previous row before each label
    % start, last null class ends at last data row
    dataset.labels_and_nullclass{c}(1:2:end,2:3) = [[1; dataset.labels{c}(:,3) + 1] [dataset.labels{c}(:,2) - 1; size(dataset.data,1)]];
    
    if ~isempty(find(diff(reshape([find(starting_logicals) find(ending_logicals)]', 2*size([find(starting_logicals) find(ending_logicals)],1), 1)) < 0))
        error('Label retrieval went wrong!');
    end
    
    % build label image
    for k = 1:size(dataset.labels{c},1)
        dataset.label_image{c}(dataset.label_id2img_coord{c}(dataset.labels{c}(k,1))) = dataset.label_image{c}(dataset.label_id2img_coord{c}(dataset.labels{c}(k,1))) + 1;
    end
end

function empty_axes(handles)
set(gcf,'CurrentAxes',handles.label_image);
set(gca, 'XTickLabel', []);
saved_handle = get(handles.label_image, 'ButtonDownFcn');
cla;
set(gca, 'XTick', []);
set(gca, 'YTick', []);
set(handles.label_image, 'ButtonDownFcn', saved_handle);

set(gcf,'CurrentAxes',handles.label_histogram);
cla;
set(gca, 'XTick', []);
set(gca, 'YTick', []);

set(handles.list_labels, 'Value', []);
set(handles.list_labels, 'String', {});



function fill_gui(handles, column)
cm = [[ones(40, 1); (1:-0.04167:0)'] [ones(16, 1); (1:-0.04167:0)'; zeros(24,1)] [1; (.5:-.0334:0)'; zeros(48,1)]];

set(gcf,'CurrentAxes',handles.label_image);
saved_handle = get(gca, 'ButtonDownFcn');
colormap(cm);
h = imagesc(handles.dataset.label_image{column});
set(handles.label_image, 'ButtonDownFcn', saved_handle);
set(h, 'ButtonDownFcn', saved_handle);
set(gca, 'XTick', 1:length(values(handles.dataset.label_id2name{column})));
set(gca, 'XTickLabel', keys(handles.dataset.label_id2name{column}));
%xticklabel_rotate(1:length(keys(handles.dataset.label_id2name{column})), 90, keys(handles.dataset.label_id2name{column}));
colorbar('location','northoutside');

set(handles.list_sensor_channels, 'String', handles.dataset.data_desc);

% build list of instances including null class CHECK
string_list = {};
for k = 1:size(handles.dataset.labels_and_nullclass{column},1)
    if handles.dataset.labels_and_nullclass{column}(k,1) == 0 % null class label:
        string_list{end+1} = [num2str(k) ': NULL class, start = ' num2str(handles.dataset.timestamps(handles.dataset.labels_and_nullclass{column}(k, 2), 1))];
    else
        string_list{end+1} = [num2str(k) ': ' handles.dataset.label_id2name{column}(handles.dataset.labels_and_nullclass{column}(k, 1)) ', start = ' num2str(handles.dataset.timestamps(handles.dataset.labels_and_nullclass{column}(k, 2), 1))];
    end
end
set(handles.list_original_stream, 'Value', []);
set(handles.list_original_stream, 'String', string_list);


% --- Executes just before label_panorama is made visible.
function label_panorama_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to label_panorama (see VARARGIN)

% Choose default command line output for label_panorama
handles.output = hObject;
handles.data_loaded = 0;
handles.sensor_scope = 0;
handles.times_to_frame_conversion = .1; % specify here number of seconds per video frame to send a frame number to the labeling tool
handles.udp = udp('127.0.0.1',5555);
fopen(handles.udp);

empty_axes(handles);

root_file = fopen('root.txt');
root_folders = {};

root = fgetl(root_file);
while ischar(root)
    root_folders{end+1} = root;
    root = fgetl(root_file);
end
fclose(root_file);
if length(root_folders) > 0
    set(handles.text_root_folder, 'String', root_folders{1});
    set(handles.pop_root_folders, 'String', root_folders);
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes label_panorama wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = label_panorama_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in list_labels.
function list_labels_Callback(hObject, eventdata, handles)
% hObject    handle to list_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns list_labels contents as cell array
%        contents{get(hObject,'Value')} returns selected item from list_labels
handles.sensor_scope = 0;
instance_set = get(hObject, 'Value');
if get(handles.btn_toggle_labeling_tool, 'Value') && length(instance_set) == 1
    time = handles.dataset.timestamps(handles.listed_label_subset(instance_set, end-1) : handles.listed_label_subset(instance_set, end), 1);
    time = round(time(1) / handles.times_to_frame_conversion); % / 100 to go from milliseconds to frames
    fwrite(handles.udp, [int2str(time) char(32 * ones(1,15 - length(int2str(time))))]); % to fill the whole buffer (which is 16 bytes)
end
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function list_labels_CreateFcn(hObject, eventdata, handles)
% hObject    handle to list_labels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_dataset.
function btn_dataset_Callback(hObject, eventdata, handles)
% hObject    handle to btn_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
root = get(handles.text_root_folder, 'String');
if isempty(root)
    root = 'C:/';
    set(handles.text_root_folder, 'String', root);
    warning('MATLAB:btn_load_Callback:no_root', 'Using C:/ as default root folder!');
end

[FileName,PathName,FilterIndex] = uigetfile({'*.dat;*.txt', 'Dataset and Legend Files'}, 'Load Dataset and Legend Files', root, 'MultiSelect','on');
if iscell(FileName)
    if length(FileName) ~= 3
        msgbox('One data file and two legend files should be selected!');
    else
        empty_axes(handles);
        handles.dataset = load_dataset(PathName, FileName);
        fill_gui(handles, 1);
        set(handles.pop_labeltrack, 'Value', []);
        set(handles.pop_labeltrack, 'String', handles.dataset.label_columns_desc);
        set(handles.pop_labeltrack, 'Value', 1);
        set(handles.txt_dataset, 'String', handles.dataset.data_file);
        handles.data_loaded = 1;
        guidata(hObject, handles);
    end
end


% --- Executes on button press in btn_reload_dataset.
function btn_reload_dataset_Callback(hObject, eventdata, handles)
% hObject    handle to btn_reload_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.txt_legend, 'String');
if ~exist([handles.dataset.path file], 'file')
    msgbox('File not found!');
else
    empty_axes(handles);
    handles.dataset = load_dataset(handles.dataset.path, {file handles.dataset.column_legend_file handles.dataset.label_legend_file});
    fill_gui(handles, 1);
    set(handles.pop_labeltrack, 'Value', []);
    set(handles.pop_labeltrack, 'String', handles.dataset.label_columns_desc);
    set(handles.pop_labeltrack, 'Value', 1);

    handles.data_loaded = 1;
    guidata(hObject, handles);
end
set(hObject, 'Enabled', 0.0);


function txt_dataset_Callback(hObject, eventdata, handles)
% hObject    handle to txt_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_dataset as text
%        str2double(get(hObject,'String')) returns contents of txt_dataset as a double
set(handles.btn_reload_dataset, 'Enabled', 1.0);

% --- Executes during object creation, after setting all properties.
function txt_dataset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_dataset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on mouse press over axes background.
function label_image_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to label_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.data_loaded == 0
    return;
end

column = get(handles.pop_labeltrack, 'Value');

clicked_label = get(gca, 'CurrentPoint');
clicked_label = round(clicked_label(1,1));
clicked_label = handles.dataset.label_img_coord2id{column}(clicked_label);

% dataset.labels is a matrix with following columns: 1 = ID, 2 = row in data
% matrix where label starts, 3 = row in data matrix where label ends
indices_cur_label = handles.dataset.labels{column}(:,1) == clicked_label;
if sum(indices_cur_label) == 0
    set(handles.text_last_event, 'String', 'No labels corresponding to the selected criteria');
    set(handles.list_labels, 'Value', []);    
    set(handles.list_labels, 'String', {});
    handles.listed_label_subset = [];
    set(gcf,'CurrentAxes',handles.label_histogram);
    cla;
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);    
else
    string_list = {};
    indices_cur_label = find(indices_cur_label);
    for k = 1:length(indices_cur_label)
        string_list{end+1} = [num2str(k) ': start = ' num2str(handles.dataset.timestamps(handles.dataset.labels{column}(indices_cur_label(k), 2), 1)) '; length = ' ...
            num2str(handles.dataset.timestamps(handles.dataset.labels{column}(indices_cur_label(k), 3), 1) - handles.dataset.timestamps(handles.dataset.labels{column}(indices_cur_label(k), 2), 1))];
    end
    set(handles.list_labels, 'Value', []);
    set(handles.list_labels, 'String', string_list);
    set(handles.text_last_event, 'String', ['Label = ''' handles.dataset.label_id2name{column}(clicked_label) '''; occurrences: ' num2str(length(string_list))]);

    set(gcf,'CurrentAxes',handles.label_histogram);
    hist(handles.dataset.timestamps(handles.dataset.labels{column}(indices_cur_label, 3), 1) - handles.dataset.timestamps(handles.dataset.labels{column}(indices_cur_label, 2), 1));
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor','r','EdgeColor','w');
    handles.listed_label_subset = handles.dataset.labels{column}(indices_cur_label, :);
end

handles.sensor_scope = 0;
guidata(hObject, handles);


% --- Executes on selection change in list_sensor_channels.
function list_sensor_channels_Callback(hObject, eventdata, handles)
% hObject    handle to list_sensor_channels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns list_sensor_channels contents as cell array
%        contents{get(hObject,'Value')} returns selected item from list_sensor_channels
handles.sensor_scope = 0;
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function list_sensor_channels_CreateFcn(hObject, eventdata, handles)
% hObject    handle to list_sensor_channels (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_signals.
function btn_signals_Callback(hObject, eventdata, handles)
% hObject    handle to btn_signals (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if isempty(get(handles.list_labels, 'Value'))
    set(handles.text_last_event, 'String', 'At least one instance should be selected!');
    return;
end
if isempty(get(handles.list_sensor_channels, 'Value'))
    set(handles.text_last_event, 'String', 'At least one sensor channel should be selected!');
    return;
end
if ~isempty(get(handles.txt_ymin, 'String')) || ~isempty(get(handles.txt_ymax, 'String'))
    ymin = str2double(get(handles.txt_ymin, 'String'));
    ymax = str2double(get(handles.txt_ymax, 'String'));
    if ymin >= ymax
        error('Check Y axis range for signals!');
    end
end
if ~isempty(get(handles.txt_tmin, 'String')) || ~isempty(get(handles.txt_tmax, 'String'))
    tmin = str2double(get(handles.txt_tmin, 'String'));
    tmax = str2double(get(handles.txt_tmax, 'String'));
    if tmin >= tmax
        error('Check time axis range for signals!');
    end
end

if handles.sensor_scope == 0
    handles.sensor_scope = figure;
    set(gcf, 'CloseRequestFcn', {'signal_scope_Callback', hObject});
    instance_set = get(handles.list_labels, 'Value');
    for instance = 1:length(instance_set)
        signal = handles.dataset.data(handles.listed_label_subset(instance_set(instance), end-1) : handles.listed_label_subset(instance_set(instance), end), handles.dataset.data_columns(get(handles.list_sensor_channels, 'Value')));
        times = handles.dataset.timestamps(handles.listed_label_subset(instance_set(instance), end-1) : handles.listed_label_subset(instance_set(instance), end), 1);
        times = times - times(1);
        subplot(length(get(handles.list_labels, 'Value')), 1, instance);
        plot(times, signal);
        if exist('ymin', 'var')
            a = axis;
            axis([a(1) a(2) ymin ymax]);
        end
        if exist('tmin', 'var')
            a = axis;
            axis([tmin tmax a(3) a(4)]);
        end        
        xlabel('Time (ms)');
    end
else
    figure(handles.sensor_scope); % select last figure plotted
    children = get(gcf, 'Children'); % get all subfigures
    for h = 1:length(children)
        set(gcf, 'CurrentAxes', children(h));
        axis auto;
        if exist('ymin', 'var')
            a = axis;
            axis([a(1) a(2) ymin ymax]);
        end
        if exist('tmin', 'var')
            a = axis;
            axis([tmin tmax a(3) a(4)]);
        end        
    end
end

guidata(hObject, handles);

function txt_ymin_Callback(hObject, eventdata, handles)
% hObject    handle to txt_ymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_ymin as text
%        str2double(get(hObject,'String')) returns contents of txt_ymin as a double


% --- Executes during object creation, after setting all properties.
function txt_ymin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_ymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txt_ymax_Callback(hObject, eventdata, handles)
% hObject    handle to txt_ymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_ymax as text
%        str2double(get(hObject,'String')) returns contents of txt_ymax as a double


% --- Executes during object creation, after setting all properties.
function txt_ymax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_ymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txt_tmin_Callback(hObject, eventdata, handles)
% hObject    handle to txt_tmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_tmin as text
%        str2double(get(hObject,'String')) returns contents of txt_tmin as a double


% --- Executes during object creation, after setting all properties.
function txt_tmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_tmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function txt_tmax_Callback(hObject, eventdata, handles)
% hObject    handle to txt_tmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_tmax as text
%        str2double(get(hObject,'String')) returns contents of txt_tmax as a double


% --- Executes during object creation, after setting all properties.
function txt_tmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_tmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
fclose(handles.udp);
delete(hObject);


% --- Executes on button press in btn_toggle_labeling_tool.
function btn_toggle_labeling_tool_Callback(hObject, eventdata, handles)
% hObject    handle to btn_toggle_labeling_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of btn_toggle_labeling_tool

if get(hObject, 'Value')
    prompt = {'Host where the Labeling Tool is running:','Port number:'};
    dlg_title = 'UDP connection to Labeling Tool';
    num_lines = 1;
    def = {'127.0.0.1','5555'};
    answer = inputdlg(prompt, dlg_title, num_lines, def);
    handles.udp = udp(answer{1}, str2num(answer{2}));
    fopen(handles.udp);
else
    fclose(handles.udp);
end
guidata(hObject, handles);


% --- Executes on selection change in pop_root_folders.
function pop_root_folders_Callback(hObject, eventdata, handles)
% hObject    handle to pop_root_folders (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_root_folders contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_root_folders
chosen_idx = get(hObject, 'Value');
chosen_root = get(hObject, 'String');
chosen_root = chosen_root{chosen_idx};
if ~isempty(chosen_root)
    set(handles.text_root_folder, 'String', chosen_root);
end
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function pop_root_folders_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_root_folders (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function text_root_folder_Callback(hObject, eventdata, handles)
% hObject    handle to text_root_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of text_root_folder as text
%        str2double(get(hObject,'String')) returns contents of text_root_folder as a double


% --- Executes during object creation, after setting all properties.
function text_root_folder_CreateFcn(hObject, eventdata, handles)
% hObject    handle to text_root_folder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in list_original_stream.
function list_original_stream_Callback(hObject, eventdata, handles)
% hObject    handle to list_original_stream (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns list_original_stream contents as cell array
%        contents{get(hObject,'Value')} returns selected item from list_original_stream


% --- Executes during object creation, after setting all properties.
function list_original_stream_CreateFcn(hObject, eventdata, handles)
% hObject    handle to list_original_stream (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_superimpose.
function btn_superimpose_Callback(hObject, eventdata, handles)
% hObject    handle to btn_superimpose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in btn_tile.
function btn_tile_Callback(hObject, eventdata, handles)
% hObject    handle to btn_tile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in pop_labeltrack.
function pop_labeltrack_Callback(hObject, eventdata, handles)
% hObject    handle to pop_labeltrack (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pop_labeltrack contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pop_labeltrack
column = get(hObject, 'Value');
empty_axes(handles);
fill_gui(handles, column);


% --- Executes during object creation, after setting all properties.
function pop_labeltrack_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pop_labeltrack (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
