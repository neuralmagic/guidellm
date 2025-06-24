import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';

const defaultSlice = createSlice({
  name: 'model.state',
  initialState: { model: 'meta-llama/Llama-2-7b' },
  reducers: {
    setDefaultData: (state, action: PayloadAction<string>) => {
      return { ...state, model: action.payload };
    },
  },
});

export const { setDefaultData } = defaultSlice.actions;

export const store = configureStore({
  reducer: {
    default: defaultSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
