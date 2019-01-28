def dice_loss(input, target,c):
    smooth = 1.
    loss = 0.
    for i in range(3):
        imask = input[:, i ]
        tmask = target[:, i]
        intersection = (imask * tmask).sum()
           
        
        loss += (1 - ((2. * intersection + smooth) /
                             (imask.sum() + tmask.sum() + smooth)))
    return loss/c