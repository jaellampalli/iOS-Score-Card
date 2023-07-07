import './App.css'
import { useState } from 'react';
import {AppBar,Toolbar,IconButton,Button, Dialog , DialogContent, TextField,DialogActions} from '@mui/material';
import {Menu as MenuIcon} from '@mui/icons-material';
import Webcam from "react-webcam";

function App() {
  const [open, setOpen] = useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <>
      <AppBar>
        <Toolbar style={{'justifyContent':"space-between"}}>
          <IconButton
            size="large"
            edge="start"
            color="inherit"
            aria-label="menu"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
            <Button color="inherit" onClick={handleClickOpen}>Login</Button>
            <Dialog open={open} onClose={handleClose}>
            <DialogContent>
              <TextField
              autoFocus
              margin="dense"
              id="name"
              label="Email Address"
              type="email"
              fullWidth
              variant="standard"
              />
              <TextField
              autoFocus
              margin="dense"
              id="name"
              label="Password"
              type="password"
              fullWidth
              variant="standard"/>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleClose}>Cancel</Button>
              <Button onClick={handleClose}>Login</Button>
            </DialogActions>
            </Dialog>


        </Toolbar>
      </AppBar>
      <Webcam></Webcam>
    </>
  )
}

export default App
