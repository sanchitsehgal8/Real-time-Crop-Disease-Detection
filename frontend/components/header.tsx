"use client"

import { Search, Settings, Sprout } from "lucide-react"
import { Input } from "@/components/ui/input"

export function Header() {
  return (
    <div>
      <header className="bg-[#e8e4df] px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          <div className="flex items-center gap-6">
            <span className="text-[#2d5a45] font-serif text-xl italic tracking-tight">
              AgroVision
            </span>
            
            <div className="relative hidden sm:block">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#6b7280]" />
              <Input 
                type="search"
                placeholder="Search..."
                className="w-48 md:w-56 pl-9 bg-[#d9d5d0] border-0 text-[#374151] placeholder:text-[#9ca3af] rounded-full focus-visible:ring-[#2d5a45] focus-visible:ring-1"
              />
            </div>
            
            <nav className="hidden md:block">
              <a 
                href="#" 
                className="text-[#374151] hover:text-[#1f2937] transition-colors font-medium"
              >
                Dashboard
              </a>
            </nav>
          </div>

          <div className="flex items-center gap-5">
            <a 
              href="#" 
              className="text-[#374151] hover:text-[#1f2937] transition-colors text-sm font-medium hidden sm:block"
            >
              Support
            </a>
            <button className="text-[#4b5563] hover:text-[#1f2937] transition-colors">
              <Settings className="w-5 h-5" />
            </button>
            <div className="w-9 h-9 rounded-full bg-[#1f3d2d] flex items-center justify-center">
              <Sprout className="w-4 h-4 text-[#a7c4b5]" />
            </div>
          </div>
        </div>
      </header>
      <div className="h-1.5 bg-gradient-to-r from-[#1a3a2a] via-[#2d5a45] to-[#1a3a2a]" />
    </div>
  )
}
