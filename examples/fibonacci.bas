#!/usr/bin/env rwbasic
REM Prints the first few numbers from the Fibonacci sequence.
i%=1:j%=1:printi%'j%:whilej%<4000000:k%=i%+j%:i%=j%:j%=k%:printj%:endwhile
