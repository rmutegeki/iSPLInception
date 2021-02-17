function [ string_out ] = underscore_clean( string_in )
string_in(strfind(string_in, '_')) = '-';
string_out = string_in;
end