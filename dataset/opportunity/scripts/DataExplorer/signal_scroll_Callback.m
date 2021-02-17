function signal_scroll_Callback(hObject, eventdata, data)
set(data.h_panel, 'position', [0 1-data.instance_number-get(hObject, 'value') 1 data.instance_number]);
