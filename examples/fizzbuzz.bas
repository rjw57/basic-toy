#!/usr/bin/env rwbasic
REM Simple fizzbuzz example.
FOR N%=1 TO 40
  FIZZ%=N% MOD 3=0:BUZZ%=N% MOD 5=0
  CASE TRUE OF
    WHEN FIZZ% AND BUZZ%: PRINT "Fizzbuzz ";
    WHEN FIZZ%: PRINT "Fizz ";
    WHEN BUZZ%: PRINT "Buzz ";
    OTHERWISE: PRINT N%;" ";
  ENDCASE
NEXT
PRINT
