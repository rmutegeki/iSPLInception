function signal_scope_Callback(hObject, eventdata, guiobject)
handles = guidata(guiobject);
handles.sensor_scope = 0;
guidata(guiobject, handles);
delete(hObject); % use "delete" instead of "close" to prevent the callback for "close" being called recursively!