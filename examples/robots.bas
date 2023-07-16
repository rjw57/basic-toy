FOR N%=1 TO 10
  IF N% = 1 THEN
    PRINT "There is ";
  ELSE
    PRINT "There are ";
  ENDIF
  PRINT N% " robot";
  IF N% <> 1 THEN PRINT "s" ELSE PRINT
NEXT
