import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:5000"
});

export const predictSong = (formData) => {
  return API.post("/predict", formData);
};
