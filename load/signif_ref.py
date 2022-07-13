import numpy as np

def signif_ref(scale, signif):
   scale = np.array(scale)
   sigma = 2
   print("This is the current scaling factor: " + str(scale))
   print("This is consequential significance: " + str(signif))
   if signif == 0:
         return scale
   elif signif < sigma:
      print("----------------------------------------")
      print("this is the old scale: " + str(scale))
      print("this is the significance: " + str(signif))
      print("this is the sigma: " + str(sigma))

      if sigma - signif < 0.1:
         #scale = scale*sigma*(1/signif) #scale + (sigma - signif)*scale/5
         scale = np.multiply(scale,scale*sigma*(1/signif))
      else:
         #scale = scale*sigma*(1/signif)
         scale = np.multiply(scale, scale * sigma * (1 / signif))
      print("this is the new scale: " + str(scale))
      print("----------------------------------------")
      print("Acording to significance I have chosen to increase the scaling")
      print("This is the new scaling factor: " + str(scale))
      return scale.tolist()
   else:
      print("----------------------------------------")
      print("this is the old scale: " + str(scale))
      print("this is the significance: " + str(signif))
      print("this is the sigma: " + str(sigma))

      
      if signif - sigma < 0.1:
         scale = np.multiply(scale, scale * sigma * (1 / signif)) #scale + (sigma - signif)*scale/5
      else:
         scale = np.multiply(scale, scale * sigma * (1 / signif))
      print("this is the new scale: " + str(scale))
      print("----------------------------------------")
      print("Acording to significance I have chosen to decrease the scaling")
      print("This is the new scaling factor: " + str(scale))
      return scale.tolist()
