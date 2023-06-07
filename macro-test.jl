macro conditional(b)
  greeting = b ? :(
    println("True: hello")     
  ) : :(
    println("False: hi")  
  )
  :(
    function greeting()
      println("I will now greet you")
      $greeting
    end
  )
end

@conditional(false)

greeting()


